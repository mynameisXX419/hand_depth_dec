#!/bin/bash
# 进入第一个程序路径并运行 python 脚本
cd /home/ljy/project/hand_dec || exit
/home/ljy/anaconda3/envs/mp_hands/bin/python hand_dec_1.py &
PID1=$!  # 保存 hand_dec_1.py 的进程号

# 启动 Qt 应用程序
cd /home/ljy/project/qt_press_image/pressprogress1/appdesign_4/build || exit
./AppDesign &
PID2=$!  # 保存 AppDesign 的进程号

echo "✅ hand_dec_1.py (PID=$PID1) 和 AppDesign (PID=$PID2) 已启动"

# 捕获 Ctrl+C 信号时同时关闭两个程序
trap "echo '⏹️ 正在关闭...'; kill $PID1 $PID2 2>/dev/null; exit" SIGINT SIGTERM

# 等待任意一个进程退出
wait -n

# 当任意一个退出后，自动关闭另一个
kill $PID1 $PID2 2>/dev/null
echo "🧹 两个程序都已结束"
