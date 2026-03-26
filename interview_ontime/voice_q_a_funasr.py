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

from voice_hot_word import get_interview_vocabulary

# 加载环境变量
load_dotenv()

# ===================== 面试场景配置 =====================
CHUNK_SIZE = 3200  # 建议使用 3200 (约 100ms)
LLM_BUFFER_SIZE = 3  # LLM 上下文滑动窗口大小
MAX_CONCURRENT_QUESTIONS = 5  # 最大并发处理的问题数

text_queue = queue.Queue()
is_running = True
active_llm_threads = []
llm_thread_lock = threading.Lock()

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
        print(f"🌐 面试助手已就绪！请在浏览器打开：http://localhost:{port}/monitor.html")

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
        print("❌ 错误：未找到 DASHSCOPE_API_KEY")
        sys.exit(1)

    # 设置官方推荐的接口地址
    dashscope.base_http_api_url = os.getenv('DASHSCOPE_URL')
    dashscope.base_websocket_api_url = os.getenv('DASHSCOPE_WS_URL')

    zhipu_api_key = os.getenv('ZHIPUAI_API_KEY')
    if not zhipu_api_key:
        print("❌ 错误：未找到 ZHIPUAI_API_KEY")
        sys.exit(1)
    return zhipu_api_key


# ===================== 动态热词管理 =====================
def create_dynamic_vocabulary():
    """动态创建专属面试热词表"""
    global vocab_service, global_vocab_id

    # 🔥 从独立配置文件加载热词
    my_vocabulary = get_interview_vocabulary()

    print(f"📚 已加载 {len(my_vocabulary)} 个面试热词")

    print("⏳ 正在向阿里云注册动态热词表，提升专业术语识别率...")
    vocab_service = VocabularyService()
    try:
        global_vocab_id = vocab_service.create_vocabulary(
            prefix='hotWord',
            target_model=os.getenv('ASR_Model'),
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
        print(f"⚠️ 热词表创建失败，将使用无热词模式继续运行。错误：{e}")
        return None


def cleanup_vocabulary():
    """清理阿里云端的临时热词表"""
    global vocab_service, global_vocab_id
    if vocab_service and global_vocab_id:
        try:
            vocab_service.delete_vocabulary(global_vocab_id)
            print(f"🗑️ 已成功清理云端临时热词表 (ID: {global_vocab_id})")
        except Exception as e:
            print(f"⚠️ 清理热词表失败：{e}")


# ===================== 全局音频变量 =====================
mic = None
stream = None


# ===================== 单个问题的 LLM 解答线程 =====================
def answer_question(question_text, question_number, zhipu_api_key):
    """为单个问题提供完整答案的独立线程"""
    print(f"\n💡 开始解答问题 #{question_number}: {question_text[:50]}...")

    client = ZhipuAiClient(api_key=zhipu_api_key)

    messages = [
        {"role": "system",
         "content": "你是一个专业的面试助手。请用简洁、清晰的语言回答问题，重点突出核心回答要点，避免冗长解释。"},
        {"role": "user", "content": f"请回答这个问题：{question_text}"}
    ]

    try:
        response = client.chat.completions.create(
            model=os.getenv('ZHIPUAI_MODEL'),
            messages=messages,
            thinking={"type": "disabled"},
            stream=True,
            max_tokens=2048,
            temperature=0.5
        )

        final_answer = ""

        # 流式获取回复
        for chunk in response:
            if chunk.choices[0].delta.content:
                final_answer += chunk.choices[0].delta.content

        final_answer = final_answer.strip()

        if final_answer:
            current_time = get_current_time_str()
            with open("LLM_response.txt", "a", encoding="utf-8") as f:
                f.write(f"❓ 问题 #{question_number} [{current_time}]:\n{question_text}\n\n")
                f.write(f"💡 参考答案 [{current_time}]:\n{final_answer}\n")
                f.write("=" * 60 + "\n\n")

            print(f"✅ 问题 #{question_number} 解答完成")

    except Exception as e:
        print(f"\n❌ 问题 #{question_number} 解答异常：{e}")


# ===================== 修复后的问题检测与分析线程 =====================
def question_detector(zhipu_api_key):
    """检测文本中是否包含问题，发现后立即启动解答线程"""
    print("🔍 问题检测线程已启动")
    client = ZhipuAiClient(api_key=zhipu_api_key)
    context_buffer = []
    question_counter = 0
    while is_running:
        try:
            item = text_queue.get(timeout=1)
        except queue.Empty:
            continue

        final_text = item['text']
        stripped_text = final_text.strip()
        # 加入上下文滑动窗口
        context_buffer.append(final_text)
        if len(context_buffer) > LLM_BUFFER_SIZE:
            context_buffer.pop(0)
        context_str = "\n".join(context_buffer)

        is_valid_question = False
        target_question_text = final_text  # 锁定真正的问题文本，避免答非所问

        # ========== 第一分支：带问号的问题检测（修复中英文问号+空白处理） ==========
        if stripped_text.endswith(("?", "？")):
            print(f"\n🔎 检测到带问号的句子，正在校验是否为面试问题...")
            detect_messages = [
                {"role": "system",
                 "content": "你是专业的面试问题检测助手。判断以下文本的核心内容，是否是面试官向面试者提出的、需要回答的新问题。如果是，仅输出「是」；如果不是（比如是回答、陈述、反问、闲聊等），仅输出「否」。禁止输出任何其他内容。"},
                {"role": "user", "content": f"文本：{context_str}"}
            ]
            try:
                detect_response = client.chat.completions.create(
                    model=os.getenv('ZHIPUAI_MODEL'),
                    messages=detect_messages,
                    thinking={"type": "disabled"},
                    stream=False,
                    max_tokens=10,
                    temperature=0.1
                )
                detect_result = detect_response.choices[0].message.content.strip()
                # 严格校验，只有明确是问题才通过
                if "是" in detect_result and "否" not in detect_result:
                    is_valid_question = True
                    target_question_text = final_text
                    context_buffer.clear()  # 检测到问题立即清空buffer，避免历史污染
            except Exception as e:
                print(f"\n❌ 带问号问题检测异常：{e}")

        # ========== 第二分支：隐含问题检测（仅当buffer满且未触发第一分支时执行） ==========
        elif len(context_buffer) >= LLM_BUFFER_SIZE and not is_valid_question:
            print(f"\n🔎 上下文缓冲区已满，检测是否有隐含面试问题...")
            detect_messages = [
                {"role": "system",
                 "content": "你是专业的面试问题检测助手。判断以下文本的最后一句话，是否是面试官向面试者提出的、需要回答的新问题。如果是，仅输出「是」；如果不是（比如是回答、陈述、闲聊、已回答过的问题等），仅输出「否」。禁止输出任何其他内容。"},
                {"role": "user", "content": f"文本：{context_str}"}
            ]
            try:
                detect_response = client.chat.completions.create(
                    model=os.getenv('ZHIPUAI_MODEL'),
                    messages=detect_messages,
                    thinking={"type": "disabled"},
                    stream=False,
                    max_tokens=10,
                    temperature=0.1
                )
                detect_result = detect_response.choices[0].message.content.strip()
                if "是" in detect_result and "否" not in detect_result:
                    is_valid_question = True
                    target_question_text = final_text
                    context_buffer.clear()  # 检测到问题立即清空buffer
            except Exception as e:
                print(f"\n❌ 隐含问题检测异常：{e}")

        # ========== 确认有效问题，启动解答线程 ==========
        if is_valid_question:
            with llm_thread_lock:
                # 清理已完成的线程
                active_llm_threads[:] = [t for t in active_llm_threads if t.is_alive()]
                # 并发数限制
                if len(active_llm_threads) >= MAX_CONCURRENT_QUESTIONS:
                    print(f"⚠️ 已达到最大并发问题数 ({MAX_CONCURRENT_QUESTIONS}),跳过当前问题")
                    continue
                question_counter += 1
                print(f"\n🎯 检测到有效问题 #{question_counter}: {target_question_text}")
                # 写入问题日志
                current_time = get_current_time_str()
                with open("LLM_voice_txt.txt", "a", encoding="utf-8") as f:
                    f.write(f"🎯 问题 #{question_counter} [{current_time}]: {target_question_text}\n")
                # 启动解答线程，传入正确的问题文本
                answer_thread = threading.Thread(
                    target=answer_question,
                    args=(target_question_text, question_counter, zhipu_api_key),
                    daemon=True
                )
                answer_thread.start()
                active_llm_threads.append(answer_thread)

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
        print('🎤 开始实时监听，ASR 结果写入 LLM_voice_txt.txt（按 Ctrl+C 停止）...')

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
        print(f'\n❌ 识别发生错误：{message.message}')
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
                sys.stdout.write(f'\r🔍 实时听写：{text}')
                sys.stdout.flush()


# ===================== 新增：前端实时更新推送 =====================
update_queue = queue.Queue()  # 用于推送实时更新到前端


def stream_to_frontend(content_type, data):
    """将内容流式推送到前端"""
    update_queue.put({
        'type': content_type,
        'data': data,
        'timestamp': get_current_time_str()
    })


def sse_handler():
    """SSE 服务器线程，持续推送更新到前端"""

    class SSEHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_GET(self):
            if self.path == '/stream':
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('Connection', 'keep-alive')
                self.end_headers()

                try:
                    while is_running:
                        try:
                            update = update_queue.get(timeout=0.5)
                            event_data = f"event: {update['type']}\ndata: {update['data']}|||{update['timestamp']}\n\n"
                            self.wfile.write(event_data.encode('utf-8'))
                        except queue.Empty:
                            continue
                except (BrokenPipeError, ConnectionResetError):
                    pass

    class ReusableTCPServer(socketserver.TCPServer):
        allow_reuse_address = True

    try:
        httpd = ReusableTCPServer(("0.0.0.0", 8849), SSEHandler)
        print(f"📡 SSE 流式服务已启动。")
        server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        server_thread.start()
    except Exception as e:
        print(f"⚠️ SSE 服务启动失败：{e}")


def send_audio_from_mic(recognition):
    while is_running:
        if stream and stream.is_active():
            try:
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                recognition.send_audio_frame(data)
            except Exception as e:
                print(f'\n❌ 音频发送异常：{e}')
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

    # 🔥 新增：启动 SSE 流式推送服务
    sse_handler()

    # 自动启动本地 Web 服务器 (8848 端口)
    start_local_server(int(os.getenv('LOCAL_PORT')))

    # 2. 动态生成热词表
    vocab_id = create_dynamic_vocabulary()

    # 2. 启动问题检测线程
    detector_thread = threading.Thread(target=question_detector, args=(zhipu_key,), daemon=True)
    detector_thread.start()

    # 3. 实例化 FunASR 识别器，挂载动态热词
    recognition = Recognition(
        model=os.getenv("ASR_Model"),
        format='pcm',
        sample_rate=16000,
        semantic_punctuation_enabled=True,
        vocabulary_id=vocab_id if vocab_id else None,
        callback=FunASRCallback()
    )

    def handle_exit(sig, frame):
        global is_running
        print('\n🛑 接收到退出信号，正在清理资源...')

        # 等待所有活跃的解答线程完成 (最多等待 5 秒)
        print("⏳ 等待正在进行的解答完成...")
        with llm_thread_lock:
            for thread in active_llm_threads:
                if thread.is_alive():
                    thread.join(timeout=5.0)

        # 发送结束事件
        stream_to_frontend('shutdown', '服务已关闭')

        is_running = False
        recognition.stop()
        cleanup_vocabulary()  # 退出时执行热词表云端清理
        print("✅ 资源清理完成，再见!")

    signal.signal(signal.SIGINT, handle_exit)

    recognition.start()

    try:
        send_audio_from_mic(recognition)
    except Exception as e:
        print(f"主循环异常：{e}")
    finally:
        is_running = False
        recognition.stop()
        cleanup_vocabulary()  # 确保异常中断也能清理词表


if __name__ == '__main__':
    main()

