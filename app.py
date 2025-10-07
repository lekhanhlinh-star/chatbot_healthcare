print("Import library")
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import os
import whisper
import torch
# from TTS.api import TTS
from rag_inference import *
from opencc import OpenCC
import uuid
import edge_tts
import base64
import random
import time
from typing import Optional

print("Finish import")
myuuid = uuid.uuid4()
app = FastAPI()

# Configure CORS origins
cors_origins = os.environ.get("CORS_ORIGINS", "https://chatbot-healthcare-ui.vercel.app")
# Convert string to list if multiple origins
if isinstance(cors_origins, str):
    if "," in cors_origins:
        cors_origins = [origin.strip() for origin in cors_origins.split(",")]
    else:
        cors_origins = [cors_origins]

# Custom CORS middleware to prevent duplicate headers
@app.middleware("http")
async def cors_handler(request, call_next):
    # Handle preflight requests
    if request.method == "OPTIONS":
        origin = request.headers.get("origin")
        if origin in cors_origins:
            return JSONResponse(
                content={},
                headers={
                    "access-control-allow-origin": origin,
                    "access-control-allow-credentials": "true",
                    "access-control-allow-methods": "GET, POST, OPTIONS",
                    "access-control-allow-headers": "Content-Type, Authorization",
                }
            )
    
    response = await call_next(request)
    
    # Add CORS headers to actual responses
    origin = request.headers.get("origin")
    if origin and origin in cors_origins:
        response.headers["access-control-allow-origin"] = origin
        response.headers["access-control-allow-credentials"] = "true"
        response.headers["access-control-allow-methods"] = "GET, POST, OPTIONS"
        response.headers["access-control-allow-headers"] = "Content-Type, Authorization"
    
    return response

UPLOAD_FOLDER = "temp"
AUDIO_CLONE = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
#init model here
print("init model")
asr_model = whisper.load_model(r"medium")
# device = "cuda" if torch.cuda.is_available() else "cpu"
# tts = TTS(model_name="tts_models/zh-CN/baker/tacotron2-DDC-GST").to(device)
cc = OpenCC('s2twp')
print("Finish init model")
def llm_inference(user_query):
    # gallary_cache = []
    
    question = user_query
    
    answer = qa_chain.invoke(question)

    # retrieved = compression_retriever.invoke(question)

    metadata_template = '''
    網頁來源:
    ({file_name})
    
    原始檔案內容:
    {content}
    --------------------------------
    '''

    # retrieved_text = ''
    
    # for i in retrieved:
    #     # print(repr(i))
    #     image_path = './split_png/' + str(i.metadata.get('page')) + '.png'
    #     retrieved_text += metadata_template.format(file_name='https://www.vtscada.com/help' + i.metadata['source'][3:], content=i.page_content)

    
    return answer

def load_questions():
    with open('questions.txt', 'r', encoding='utf-8') as f:
        questions = [line.strip() for line in f if line.strip()]
    return questions

@app.get("/ping")
async def ping():
    return {"status": "healthy"}

@app.post("/upload")
async def upload_audio(audio: UploadFile = File(...)):
    if not audio:
        raise HTTPException(status_code=400, detail="沒有音訊檔案")

    # Secure filename
    filename = audio.filename
    if filename:
        # Remove path components and keep only filename
        filename = os.path.basename(filename)
    else:
        filename = f"audio_{uuid.uuid4()}.wav"
    
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    # Save uploaded file
    with open(filepath, "wb") as buffer:
        content = await audio.read()
        buffer.write(content)

    result = asr_model.transcribe(filepath, language="zh")
    
    # Bạn có thể gọi ASR để chuyển đổi giọng nói thành văn bản ở đây
    # ví dụ: result = my_asr(filepath)
    answer = cc.convert(result['text'])
    return answer

@app.post("/ask")
async def ask(
    question: str = Form(...),
    role: Optional[str] = Form("unknown"),
    responseWithAudio: Optional[str] = Form("false")
):
    print(f"Question: {question}, Role: {role}, Audio: {responseWithAudio}")
    if not question:
        raise HTTPException(status_code=400, detail="請輸入問題")

        # Gọi mô hình để trả lời câu hỏi ở đây
        # ví dụ: answer = my_model.answer(question)
    start_time = time.time()
    answer = llm_inference(question)
    end_time = time.time() - start_time
    print("LLM:",end_time)
    start_time = time.time()
    answer = cc.convert(answer)
    end_time = time.time() - start_time
    print("Convert:",end_time)
    start_time = time.time()
    if responseWithAudio == "true":
        if role == "doctor":
            voices = "zh-CN-YunyangNeural"
        else:
           voices = "zh-CN-XiaoxiaoNeural"
        myuuid = uuid.uuid4()
        audio_name = str(myuuid) + '.mp3'
        filepath = os.path.join(UPLOAD_FOLDER, audio_name)
        tts = edge_tts.Communicate(
                text=answer,
                voice=voices)
        await tts.save(filepath)
        # return jsonify({"answer": answer, "audio": audio, "filepath": filepath})
        with open(filepath, 'rb') as audio_file:
            audio_data = audio_file.read()
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        end_time = time.time() - start_time
        print("Audio:",end_time)
        return {"answer": answer, "audio_base64": audio_base64}
    else:
        return {"answer": answer}

if __name__ == "__main__":
    import uvicorn
    # Get port from environment variable, default to 5012
    port = int(os.environ.get("PORT", 80))
    
    print(f"Starting FastAPI server on port {port}")
    print(f"Health check endpoint available at /ping")
    
    uvicorn.run(app, host="0.0.0.0", port=port)
