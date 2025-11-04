#!/bin/bash
# ============================================
# run_all.sh — 本机桌面稳定版（无需设置 DISPLAY）
# ============================================

# 清理旧 socket
rm -f /tmp/press_event.sock

# --- 启动 Qt 程序 ---
cd /home/ljy/project/qt_press_image/pressprogress1/appdesign_4/build || exit
(./AppDesign > /tmp/appdesign.log 2>&1 &) 
PID_APP=$!
echo "🚀 已启动 AppDesign (PID=$PID_APP)，等待 10 秒初始化..."
sleep 10

# --- 启动 Python ---
cd /home/ljy/project/hand_dec || exit
nohup /home/ljy/anaconda3/envs/mp_hands/bin/python hand_dec_2.py > /tmp/hand_dec.log 2>&1 &
PID_HAND=$!
nohup /home/ljy/anaconda3/envs/mp_hands/bin/python listener_fusion.py > /tmp/listener_fusion.log 2>&1 &
PID_LISTENER=$!
echo "✅ hand_dec_2.py (PID=$PID_HAND) 与 listener_fusion.py (PID=$PID_LISTENER) 已启动"

# --- 捕获 Ctrl+C ---
trap "echo '⏹️ 正在关闭所有进程...'; kill $PID_APP $PID_HAND $PID_LISTENER 2>/dev/null; exit" SIGINT SIGTERM

wait -n
kill $PID_APP $PID_HAND $PID_LISTENER 2>/dev/null
echo "🧹 所有程序已结束。"
