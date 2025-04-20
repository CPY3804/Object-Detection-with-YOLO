import cv2
from ultralytics import YOLO

# 加載 YOLO 模型
model = YOLO("yolo11n.pt")  # 替換為適合的 YOLO 模型路徑

def draw_direction_hint(frame, move_command):
    """
    在畫面上繪製提示用戶的移動方向。
    - frame: 當前影像
    - move_command: "left", "right", "stop", "forward", or "analyze"
    """
    h, w, _ = frame.shape
    color = (0, 255, 0)  # 預設綠色提示
    thickness = 3

    if move_command == "left":
        # 提示向左閃避
        cv2.putText(frame, "LEFT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        cv2.arrowedLine(frame, (w // 2, h // 2), (w // 2 - 100, h // 2), color, thickness, tipLength=0.5)
    elif move_command == "right":
        # 提示向右閃避
        cv2.putText(frame, "RIGHT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        cv2.arrowedLine(frame, (w // 2, h // 2), (w // 2 + 100, h // 2), color, thickness, tipLength=0.5)
    elif move_command == "stop":
        # 提示停止
        cv2.putText(frame, "STOP", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)  # 紅色警告
    elif move_command == "analyze":
        # 提示分析中
        cv2.putText(frame, "ANALYZING...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)

def get_movement_command(boxes, frame_width, frame_height):
    """
    根據障礙物位置決定提示方向。
    - boxes: YOLO 檢測結果中的 Boxes 物件
    - frame_width: 畫面寬度
    - frame_height: 畫面高度
    返回 "left", "right", "stop", "forward", or "analyze"
    """
    if boxes is None or len(boxes) == 0:
        return "forward"  # 無檢測結果，繼續前進

    for box in boxes.xyxy:
        x1, y1, x2, y2 = box[:4].tolist()
        center_x = (x1 + x2) / 2

        # 根據障礙物位置決定提示
        if center_x < frame_width * 0.4 and y2 > frame_height * 0.7:
            return "right"  # 障礙物在左上角，提示向右
        elif frame_width * 0.6 < center_x  and y2 > frame_height * 0.7:
            return "left"  # 障礙物在右上角，提示向左
        elif frame_width * 0.4 < center_x < frame_width * 0.6 and y2 > frame_height * 0.7:
            return "stop"  # 障礙物在中間，提示停止
    return "analyze"  # 不確定的情況，提示分析

def main():
    # 開啟攝像頭
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("未獲取到影像，退出中...")
            break

        # YOLO 推論
        results = model(frame)

        # 根據檢測結果計算移動提示
        move_command = get_movement_command(results[0].boxes, frame.shape[1], frame.shape[0]) 

        # 在畫面上繪製檢測結果
        annotated_frame = results[0].plot()

        # 添加移動提示到畫面
        draw_direction_hint(annotated_frame, move_command)

        # 在畫面上繪製檢測結果
        annotated_frame = results[0].plot()

        # 添加十字準心到每個框框
        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = box[:4].tolist()
            center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)

            # 繪製十字準心
            color = (255, 255, 255)  # 十字準心顏色 (黃色)
            thickness = 1  # 十字線的厚度
            size = 15  # 十字準心大小

            # 水平線
            cv2.line(annotated_frame, (center_x - size, center_y), (center_x + size, center_y), color, thickness)
            # 垂直線
            cv2.line(annotated_frame, (center_x, center_y - size), (center_x, center_y + size), color, thickness)

        # 添加移動提示到畫面
        draw_direction_hint(annotated_frame, move_command)

        # 顯示畫面
        cv2.imshow("Obstacle Avoidance Hint", annotated_frame)

        # 按下 'q' 鍵退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("退出程式...")
            break

    cap.release()
    cv2.destroyAllWindows()

# 啟動主程式
if __name__ == "__main__":
    main()
