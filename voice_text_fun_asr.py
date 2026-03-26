# -*- coding: utf8 -*-
import logging
import os
import signal
import sys
import time
from datetime import datetime
import threading
import queue
import pyaudio
import dashscope
from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult, VocabularyService
from dotenv import load_dotenv
import http.server
import socketserver

# --- 引入官方 ZhipuAI SDK ---
from zai import ZhipuAiClient

# 加载环境变量
load_dotenv()

# ===================== 面试场景配置 =====================
CHUNK_SIZE = 3200  # 建议使用 3200 (约 100ms)
LLM_BUFFER_SIZE = 3  # LLM 上下文滑动窗口大小

text_queue = queue.Queue()
is_running = True

# 全局热词服务变量，便于退出时清理
vocab_service = None
global_vocab_id = None


# ===================== 内置本地前端服务器 =====================
def start_local_server(port=8848):
    """在后台独立线程启动 HTTP 服务，提供前端页面访问"""

    class ReusableTCPServer(socketserver.TCPServer):
        allow_reuse_address = True  # 允许端口复用

    # 🔥 核心修改：创建一个静音版的 Handler
    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # 直接 pass 掉，不再往控制台打印任何 GET 请求日志

    try:
        # 这里使用我们刚刚定义的 QuietHandler
        httpd = ReusableTCPServer(("", port), QuietHandler)
        print(f"🌐 面试助手已就绪！请在浏览器打开: http://localhost:{port}/monitor.html")

        # 将 HTTP 服务放入守护线程
        server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        server_thread.start()
    except Exception as e:
        print(f"⚠️ 本地服务器启动失败 (请检查端口 {port} 是否被占用): {e}")



def get_current_time_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-5]


def setup_logging():
    logger = logging.getLogger('dashscope')
    logger.setLevel(logging.WARNING)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.WARNING)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def init_api_keys():
    dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')
    if not dashscope.api_key:
        print("❌ 错误: 未找到 DASHSCOPE_API_KEY")
        sys.exit(1)

    # 设置官方推荐的接口地址
    dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'
    dashscope.base_websocket_api_url = 'wss://dashscope.aliyuncs.com/api-ws/v1/inference'

    zhipu_api_key = os.getenv('ZHIPUAI_API_KEY')
    if not zhipu_api_key:
        print("❌ 错误: 未找到 ZHIPUAI_API_KEY")
        sys.exit(1)
    return zhipu_api_key


# ===================== 动态热词管理 =====================
def create_dynamic_vocabulary():
    """动态创建专属面试热词表"""
    global vocab_service, global_vocab_id

    # 针对性预设的热词，可自行增删
    my_vocabulary = [
        {"text": "大语言模型", "weight": 4},
        {"text": "大模型", "weight": 4},
        {"text": "LangChain", "weight": 5},
        {"text": "LangGraph", "weight": 5},
        {"text": "RAG", "weight": 4},
        {"text": "Transformer", "weight": 4},
        {"text": "Neo4j", "weight": 5},
        {"text": "知识图谱", "weight": 5},
        {"text": "ChatBI", "weight": 5},
        {"text": "强化学习", "weight": 4},
        {"text": "Graph RAG", "weight": 4},
        {"text": "Agent", "weight": 5},
        {"text": "智能体", "weight": 3},
        {"text": "PPO", "weight": 3}
    ]

    print("⏳ 正在向阿里云注册动态热词表，提升专业术语识别率...")
    vocab_service = VocabularyService()
    try:
        global_vocab_id = vocab_service.create_vocabulary(
            prefix='hotWord',
            target_model='fun-asr-realtime',
            vocabulary=my_vocabulary
        )

        # 轮询等待热词表就绪
        while True:
            status = vocab_service.query_vocabulary(global_vocab_id)['status']
            if status == 'OK':
                print(f"✅ 专属热词表创建成功 (ID: {global_vocab_id})")
                break
            time.sleep(0.5)

        return global_vocab_id
    except Exception as e:
        print(f"⚠️ 热词表创建失败，将使用无热词模式继续运行。错误: {e}")
        return None


def cleanup_vocabulary():
    """清理阿里云端的临时热词表"""
    global vocab_service, global_vocab_id
    if vocab_service and global_vocab_id:
        try:
            vocab_service.delete_vocabulary(global_vocab_id)
            print(f"🗑️ 已成功清理云端临时热词表 (ID: {global_vocab_id})")
        except Exception as e:
            print(f"⚠️ 清理热词表失败: {e}")


# ===================== 全局音频变量 =====================
mic = None
stream = None


# ===================== 后台 LLM 处理线程 =====================
def llm_worker(zhipu_api_key):
    print("🧠 LLM 分析线程已启动 (极速模式，关闭深度思考)...")
    client = ZhipuAiClient(api_key=zhipu_api_key)
    context_buffer = []

    while is_running:
        try:
            item = text_queue.get(timeout=1)
        except queue.Empty:
            continue

        final_text = item['text']
        context_buffer.append(final_text)

        if len(context_buffer) > LLM_BUFFER_SIZE:
            context_buffer.pop(0)

        context_str = " ".join(context_buffer)

        if final_text.endswith("？") or final_text.endswith("?") or len(context_buffer) >= LLM_BUFFER_SIZE:
            # 🔥 优化1：极其严苛且简短的 Prompt
            messages = [
                {"role": "system",
                 "content": "你是一个极速的面试提词器。任务：判断文本中是否包含面试官的问题。如果有，直接输出该问题，绝对不要任何解释或寒暄。如果没有问题，只输出一个字：无。"},
                {"role": "user", "content": f"文本：\n{context_str}"}
            ]

            try:
                response = client.chat.completions.create(
                    model="glm-4.7-flash",
                    messages=messages,
                    thinking={"type": "disabled"},
                    stream=True,
                    max_tokens=1024,
                    temperature=0.3
                )

                final_answer = ""

                # 流式获取回复 (去掉了思考过程的解析)
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        final_answer += chunk.choices[0].delta.content

                final_answer = final_answer.strip()

                # 🔥 优化3：精准判断，只有真正提取到问题才写入
                if final_answer and final_answer != "无" and "未检测" not in final_answer:
                    current_time = get_current_time_str()
                    with open("LLM_response.txt", "a", encoding="utf-8") as f:
                        # 直接写入结论，不再有思考过程
                        f.write(f"🎯 核心结论 [{current_time}]:\n{final_answer}\n")
                        f.write("=" * 60 + "\n\n")

                    context_buffer.clear()

            except Exception as e:
                print(f"\n❌ LLM 解析异常: {e}")

        text_queue.task_done()


# ===================== FunASR 实时识别回调 =====================
class FunASRCallback(RecognitionCallback):
    def on_open(self) -> None:
        print('✅ 阿里云 FunASR 连接成功')
        global mic, stream
        mic = pyaudio.PyAudio()
        stream = mic.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True
        )
        print('🎤 开始实时监听，ASR结果写入 LLM_voice_txt.txt（按Ctrl+C停止）...')

    def on_close(self) -> None:
        print(f'\n🔌 FunASR 连接已关闭')
        global stream, mic
        if stream and stream.is_active():
            stream.stop_stream()
            stream.close()
        if mic:
            mic.terminate()
        stream = None
        mic = None

    def on_complete(self) -> None:
        pass

    def on_error(self, message) -> None:
        print(f'\n❌ 识别发生错误: {message.message}')
        global is_running
        is_running = False
        if 'stream' in globals() and stream and stream.is_active():
            stream.stop_stream()
            stream.close()
        sys.exit(1)

    def on_event(self, result: RecognitionResult) -> None:
        sentence = result.get_sentence()
        if 'text' in sentence:
            text = sentence['text']

            if RecognitionResult.is_sentence_end(sentence):
                sys.stdout.write('\r' + ' ' * 80 + '\r')
                sys.stdout.flush()

                current_time = get_current_time_str()
                print(f"[{current_time}] 📝 {text}")

                with open("LLM_voice_txt.txt", "a", encoding="utf-8") as f:
                    f.write(f"[{current_time}] {text}\n")

                text_queue.put({"text": text})
            else:
                sys.stdout.write(f'\r🔍 实时听写: {text}')
                sys.stdout.flush()


def send_audio_from_mic(recognition):
    while is_running:
        if stream and stream.is_active():
            try:
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                recognition.send_audio_frame(data)
            except Exception as e:
                print(f'\n❌ 音频发送异常: {e}')
                break
        else:
            time.sleep(0.1)


def init_log_files():
    """每次启动脚本时，清空上一次遗留的文本内容"""
    with open("LLM_voice_txt.txt", "w", encoding="utf-8") as f:
        f.write("")
    with open("LLM_response.txt", "w", encoding="utf-8") as f:
        f.write("")
    print("🧹 已清空历史日志文件，准备迎接新对话。")


def main():
    global is_running

    # 1. 启动时先清空历史文本文件
    init_log_files()

    setup_logging()
    zhipu_key = init_api_keys()

    # 自动启动本地 Web 服务器 (8848端口)
    start_local_server(8848)

    # 2. 动态生成热词表
    vocab_id = create_dynamic_vocabulary()

    # 2. 启动 LLM 后台处理线程
    llm_thread = threading.Thread(target=llm_worker, args=(zhipu_key,), daemon=True)
    llm_thread.start()

    # 3. 实例化 FunASR 识别器，挂载动态热词
    recognition = Recognition(
        model='fun-asr-realtime',
        format='pcm',
        sample_rate=16000,
        semantic_punctuation_enabled=True,
        vocabulary_id=vocab_id if vocab_id else None,
        callback=FunASRCallback()
    )

    def handle_exit(sig, frame):
        global is_running
        print('\n🛑 接收到退出信号，正在清理资源...')
        is_running = False
        recognition.stop()
        cleanup_vocabulary()  # 退出时执行热词表云端清理
        # sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)

    recognition.start()

    try:
        send_audio_from_mic(recognition)
    except Exception as e:
        print(f"主循环异常: {e}")
    finally:
        is_running = False
        recognition.stop()
        cleanup_vocabulary()  # 确保异常中断也能清理词表


if __name__ == '__main__':
    main()
