print("Import library")
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
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

print("Finish import")
myuuid = uuid.uuid4()
app = Flask(__name__)

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

@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        cors_origins = os.environ.get("CORS_ORIGINS", "https://chatbot-healthcare-ui.vercel.app")
        response = jsonify({})
        response.headers['Access-Control-Allow-Origin'] = cors_origins
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        return response

@app.after_request
def after_request(response):
    cors_origins = os.environ.get("CORS_ORIGINS", "https://chatbot-healthcare-ui.vercel.app")
    # Remove any existing CORS headers to prevent duplicates
    response.headers.pop('Access-Control-Allow-Origin', None)
    response.headers.pop('Access-Control-Allow-Methods', None)
    response.headers.pop('Access-Control-Allow-Headers', None)
    response.headers.pop('Access-Control-Allow-Credentials', None)
    # Set our CORS headers
    response.headers['Access-Control-Allow-Origin'] = cors_origins
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    return response

@app.route("/ping")
def ping():
    return jsonify({"status": "healthy"}), 200

@app.route("/upload", methods=["POST"])
def upload_audio():
    if "audio" not in request.files:
        return "沒有音訊檔案", 400

    file = request.files["audio"]
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    result = asr_model.transcribe(filepath, language="zh")
    
    # Bạn có thể gọi ASR để chuyển đổi giọng nói thành văn bản ở đây
    # ví dụ: result = my_asr(filepath)
    answer = cc.convert(result['text'])
    return answer

@app.route("/ask", methods=["POST"])
async def ask():
    question = request.form.get("question")
    role = request.form.get('role', 'unknown')
    responseWithAudio = request.form.get("responseWithAudio", False)
    print(f"Question: {question}, Role: {role}, Audio: {responseWithAudio}")
    if not question:
        return "請輸入問題", 400

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
        return jsonify({"answer": answer, "audio_base64": audio_base64})
    else:
        return jsonify({"answer": answer})

if __name__ == "__main__":
    # Get port from environment variable, default to 5012
    port = int(os.environ.get("PORT", 80))
    # Health check port can be the same as main port or different
    health_port = int(os.environ.get("PORT_HEALTH", port))
    
    print(f"Starting server on port {port}")
    print(f"Health check endpoint available at port {health_port}/ping")
    
    app.run(host="0.0.0.0", port=port, debug=True)
