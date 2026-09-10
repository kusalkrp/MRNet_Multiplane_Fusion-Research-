"""
TriPlane Health — ACL MRI Injury Diagnostic Inference API
FastAPI Service providing real-time multiplane knee injury predictions,
model comparisons, Grad-CAM attention explainability, and audit trails.
"""

import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

try:
    from .inference import engine, MODEL_REGISTRY, PLANES, ModelNotFoundError, ModelNotAvailableError
    from .preprocessing import (
        parse_npy_bytes,
        parse_npy_base64,
        PreprocessingError,
        validate_mri_volume
    )
    from .gradcam import generate_gradcam_for_volume, GradCAMNotSupportedError
    from .audit import log_prediction, get_recent_logs, init_db
    from .schemas import (
        HealthResponse,
        ModelsListResponse,
        ModelDetail,
        PlaneMetricInfo,
        PredictRequestJSON,
        PredictResponse,
        CompareResponse,
        PlaneLogits,
        FusionWeights,
        GradCAMRequestJSON,
        GradCAMResponse,
        HotspotCoord,
        AuditLogsResponse,
        AuditLogItem
    )
except (ImportError, ValueError):
    from inference import engine, MODEL_REGISTRY, PLANES, ModelNotFoundError, ModelNotAvailableError
    from preprocessing import (
        parse_npy_bytes,
        parse_npy_base64,
        PreprocessingError,
        validate_mri_volume
    )
    from gradcam import generate_gradcam_for_volume, GradCAMNotSupportedError
    from audit import log_prediction, get_recent_logs, init_db
    from schemas import (
        HealthResponse,
        ModelsListResponse,
        ModelDetail,
        PlaneMetricInfo,
        PredictRequestJSON,
        PredictResponse,
        CompareResponse,
        PlaneLogits,
        FusionWeights,
        GradCAMRequestJSON,
        GradCAMResponse,
        HotspotCoord,
        AuditLogsResponse,
        AuditLogItem
    )

BASE_DIR = Path(__file__).parent.parent
DATASET_VALID_DIR = BASE_DIR / "dataset" / "valid"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup & shutdown lifecycle: preload all 9 PyTorch checkpoints and 3 fusion models."""
    print("=" * 60)
    print("Starting TriPlane Health Inference API...")
    init_db()
    engine.load_all()
    loaded = [k for k, v in engine.load_status.items() if v]
    print(f"Models successfully loaded into memory: {loaded}")
    print(f"Execution device: {engine.device}")
    print("=" * 60)
    yield
    print("Shutting down TriPlane Health Inference API.")


app = FastAPI(
    title="TriPlane Health — ACL Injury Inference API",
    description=(
        "Production-grade diagnostic inference service for Knee MRI injury detection. "
        "Fuses Multi-Plane (Axial, Coronal, Sagittal) PyTorch Experts using Stage-2 Logistic Regression."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# EXCEPTION HANDLERS
# ============================================================================

@app.exception_handler(PreprocessingError)
async def preprocessing_exception_handler(request: Request, exc: PreprocessingError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": str(exc), "error_type": "PreprocessingError"}
    )


@app.exception_handler(ModelNotFoundError)
async def model_not_found_handler(request: Request, exc: ModelNotFoundError):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": str(exc), "error_type": "ModelNotFoundError"}
    )


@app.exception_handler(ModelNotAvailableError)
async def model_not_available_handler(request: Request, exc: ModelNotAvailableError):
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": str(exc), "error_type": "ModelNotAvailableError"}
    )


@app.exception_handler(GradCAMNotSupportedError)
async def gradcam_not_supported_handler(request: Request, exc: GradCAMNotSupportedError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": str(exc), "error_type": "GradCAMNotSupportedError"}
    )


# ============================================================================
# SYSTEM HEALTH & METADATA ENDPOINTS
# ============================================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def get_health():
    """Healthcheck endpoint returning system status and cached models."""
    loaded = [k for k, v in engine.load_status.items() if v]
    return HealthResponse(
        status="ok",
        models_loaded=loaded,
        device=str(engine.device),
        total_models_available=len(loaded)
    )


@app.get("/models", response_model=ModelsListResponse, tags=["System"])
async def list_models():
    """Returns available model architectures, attention support, and checkpoint metrics."""
    models_list = []
    for k, meta in engine.model_metadata.items():
        planes_dict = {
            p: PlaneMetricInfo(
                best_auc=meta["planes"].get(p, {}).get("best_auc", 0.0),
                loss_type=meta["planes"].get(p, {}).get("loss_type", "bce"),
                checkpoint=meta["planes"].get(p, {}).get("checkpoint", "")
            )
            for p in PLANES if p in meta.get("planes", {})
        }
        models_list.append(
            ModelDetail(
                model_key=k,
                name=meta["name"],
                description=meta["description"],
                has_attention=meta["has_attention"],
                paper_auc=meta["paper_auc"],
                is_loaded=meta["is_loaded"],
                planes=planes_dict
            )
        )
    return ModelsListResponse(models=models_list)


# ============================================================================
# PREDICTION ENDPOINTS (JSON & MULTIPART FORM)
# ============================================================================

def _build_predict_response(raw_res: Dict[str, Any], exam_id: Optional[str], elapsed_ms: float) -> PredictResponse:
    return PredictResponse(
        model=raw_res["model"],
        probability=raw_res["probability"],
        predicted_label=raw_res["predicted_label"],
        threshold=raw_res["threshold"],
        plane_logits=PlaneLogits(**raw_res["plane_logits"]),
        fusion_weights=FusionWeights(**raw_res["fusion_weights"]),
        has_attention=raw_res["has_attention"],
        exam_id=exam_id,
        execution_time_ms=round(elapsed_ms, 2)
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict_json(req: PredictRequestJSON):
    """
    Runs multiplane ACL injury detection for a single model family using base64 JSON payload.
    """
    t0 = time.perf_counter()
    vol_axial = parse_npy_base64(req.axial, plane_name="axial")
    vol_coronal = parse_npy_base64(req.coronal, plane_name="coronal")
    vol_sagittal = parse_npy_base64(req.sagittal, plane_name="sagittal")

    res = engine.predict(
        model_key=req.model,
        axial_vol=vol_axial,
        coronal_vol=vol_coronal,
        sagittal_vol=vol_sagittal,
        threshold=req.threshold
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Record in SQLite audit log
    log_prediction(
        model=req.model,
        probability=res["probability"],
        predicted_label=res["predicted_label"],
        plane_logits=res["plane_logits"],
        execution_time_ms=elapsed_ms,
        exam_id=req.exam_id
    )

    return _build_predict_response(res, req.exam_id, elapsed_ms)


@app.post("/predict/form", response_model=PredictResponse, tags=["Inference"])
async def predict_form(
    model: str = Form("proposed"),
    axial: UploadFile = File(..., description="Axial plane .npy file"),
    coronal: UploadFile = File(..., description="Coronal plane .npy file"),
    sagittal: UploadFile = File(..., description="Sagittal plane .npy file"),
    threshold: float = Form(0.5),
    exam_id: Optional[str] = Form(None)
):
    """
    Runs multiplane ACL injury detection using multipart file upload.
    """
    t0 = time.perf_counter()
    vol_axial = parse_npy_bytes(await axial.read(), plane_name="axial")
    vol_coronal = parse_npy_bytes(await coronal.read(), plane_name="coronal")
    vol_sagittal = parse_npy_bytes(await sagittal.read(), plane_name="sagittal")

    res = engine.predict(
        model_key=model,
        axial_vol=vol_axial,
        coronal_vol=vol_coronal,
        sagittal_vol=vol_sagittal,
        threshold=threshold
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    log_prediction(
        model=model,
        probability=res["probability"],
        predicted_label=res["predicted_label"],
        plane_logits=res["plane_logits"],
        execution_time_ms=elapsed_ms,
        exam_id=exam_id
    )

    return _build_predict_response(res, exam_id, elapsed_ms)


@app.post("/predict/compare", response_model=List[PredictResponse], tags=["Inference"])
async def predict_compare_json(req: PredictRequestJSON):
    """
    Runs all 3 model architectures live on the same case and returns comparative array.
    """
    t0 = time.perf_counter()
    vol_axial = parse_npy_base64(req.axial, plane_name="axial")
    vol_coronal = parse_npy_base64(req.coronal, plane_name="coronal")
    vol_sagittal = parse_npy_base64(req.sagittal, plane_name="sagittal")

    raw_results = engine.predict_compare(
        axial_vol=vol_axial,
        coronal_vol=vol_coronal,
        sagittal_vol=vol_sagittal,
        threshold=req.threshold
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Log the primary (proposed) model in audit trail
    for item in raw_results:
        if item["model"] == "proposed":
            log_prediction(
                model="proposed",
                probability=item["probability"],
                predicted_label=item["predicted_label"],
                plane_logits=item["plane_logits"],
                execution_time_ms=elapsed_ms,
                exam_id=req.exam_id
            )

    return [_build_predict_response(r, req.exam_id, elapsed_ms / len(raw_results)) for r in raw_results]


# ============================================================================
# GRAD-CAM VISUAL EXPLAINABILITY
# ============================================================================

@app.post("/gradcam", response_model=GradCAMResponse, tags=["Explainability"])
async def get_gradcam(req: GradCAMRequestJSON):
    """
    Generates class activation mapping (Grad-CAM) for a selected model plane & slice.
    Rejects 'custom_cnn' with 422.
    """
    if req.model == "custom_cnn":
        raise GradCAMNotSupportedError("custom_cnn has no spatial attention mechanism to extract.")

    # Determine input volume
    vol_raw = req.volume
    if not vol_raw:
        if req.plane == "axial" and req.axial:
            vol_raw = req.axial
        elif req.plane == "coronal" and req.coronal:
            vol_raw = req.coronal
        elif req.plane == "sagittal" and req.sagittal:
            vol_raw = req.sagittal

    if not vol_raw:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Missing volume data for plane '{req.plane}'."
        )

    volume = parse_npy_base64(vol_raw, plane_name=req.plane)
    expert_model = engine.get_model(req.model, req.plane)

    cam_data = generate_gradcam_for_volume(
        model=expert_model,
        model_key=req.model,
        plane=req.plane,
        volume=volume,
        slice_idx=req.slice_idx,
        target_layer_name=req.target_layer,
        device=engine.device
    )

    return GradCAMResponse(
        model=cam_data["model"],
        plane=cam_data["plane"],
        slice_idx=cam_data["slice_idx"],
        total_slices=cam_data["total_slices"],
        confidence=cam_data["confidence"],
        target_layer=cam_data["target_layer"],
        hotspot=HotspotCoord(**cam_data["hotspot"]),
        original_image=cam_data["original_image"],
        heatmap_image=cam_data["heatmap_image"],
        overlay_image=cam_data["overlay_image"]
    )


# ============================================================================
# CLINICAL CASE BROWSER & DEMO SAMPLES
# ============================================================================

@app.get("/cases", tags=["Cases"])
async def list_available_cases():
    """
    Lists sample cases available on the server with ground truth labels for instant 1-click loading.
    """
    if not DATASET_VALID_DIR.exists():
        return {"cases": []}

    # Gather ground truth CSV for ACL
    acl_csv = BASE_DIR / "dataset" / "valid-acl.csv"
    gt_map = {}
    if acl_csv.exists():
        import pandas as pd
        df = pd.read_csv(acl_csv, header=None, names=["exam_id", "label"])
        for _, row in df.iterrows():
            gt_map[str(row["exam_id"])] = int(row["label"])

    cases = []
    axial_dir = DATASET_VALID_DIR / "axial"
    if axial_dir.exists():
        for f in sorted(axial_dir.glob("*.npy")):
            eid = f.stem
            # Check other planes
            has_coronal = (DATASET_VALID_DIR / "coronal" / f"{eid}.npy").exists()
            has_sagittal = (DATASET_VALID_DIR / "sagittal" / f"{eid}.npy").exists()
            if has_coronal and has_sagittal:
                label = gt_map.get(eid, None)
                label_str = "Tear (Positive)" if label == 1 else "Normal (Intact)" if label == 0 else "Unknown"
                cases.append({
                    "exam_id": eid,
                    "ground_truth": label,
                    "ground_truth_label": label_str,
                    "is_held_out_test": True
                })

    return {"total": len(cases), "cases": cases}


@app.get("/cases/{exam_id}", tags=["Cases"])
async def get_case_volumes(exam_id: str):
    """
    Returns base64-encoded volumes for an exam ID directly from the server dataset.
    Enables instant demonstration in the UI without manual file uploads.
    """
    planes_b64 = {}
    import base64
    for plane in PLANES:
        p = DATASET_VALID_DIR / plane / f"{exam_id}.npy"
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"Plane {plane} for exam {exam_id} not found.")
        with open(p, "rb") as f:
            planes_b64[plane] = base64.b64encode(f.read()).decode("utf-8")

    return {
        "exam_id": exam_id,
        "axial": planes_b64["axial"],
        "coronal": planes_b64["coronal"],
        "sagittal": planes_b64["sagittal"],
    }


# ============================================================================
# AUDIT TRAIL ENDPOINT
# ============================================================================

@app.get("/audit-logs", response_model=AuditLogsResponse, tags=["Audit"])
async def get_audit_trail(limit: int = 50):
    """Retrieves chronological audit trail records from the SQLite database."""
    logs = get_recent_logs(limit=limit)
    items = [AuditLogItem(**l) for l in logs]
    return AuditLogsResponse(total_records=len(items), logs=items)
