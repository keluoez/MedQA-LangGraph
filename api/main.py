"""MedQA-LangGraph 演示 API 入口。

只做编排：加载 .env、CORS、挂载路由。
业务逻辑全部在原有 backend 代码与 api/adapters 内，本文件保持薄。
"""
from dotenv import load_dotenv

from api.adapters.pipeline_bridge import ROOT

load_dotenv(ROOT / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import dataset, mechanism, qa

app = FastAPI(
    title="MedQA-LangGraph Demo API",
    description="多智能体生物医学 QA 数据集生成框架 · 演示接口（mock 默认 / live 可切换）",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(dataset.router)
app.include_router(qa.router)
app.include_router(mechanism.router)


@app.get("/api/health")
def health():
    return {"status": "ok"}
