import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from ultralytics import YOLO


class YoloDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        # 加载预训练的 YOLOv8 模型
        self.model = YOLO(r'D:\code\python\Graduation_project\weights\best.pt')

    def initUI(self):
        layout = QVBoxLayout()

        self.load_button = QPushButton('加载图像', self)
        self.load_button.clicked.connect(self.load_image)
        layout.addWidget(self.load_button)

        self.image_label = QLabel(self)
        layout.addWidget(self.image_label)

        self.setLayout(layout)
        self.setWindowTitle('YOLO 垃圾检测系统')
        self.show()

    def load_image(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, '选择图像文件', '', '图像文件 (*.jpg *.jpeg *.png)')
        if file_path:
            try:
                # 读取图像
                image = cv2.imread(file_path)
                if image is not None:
                    # 使用 YOLO 模型进行检测
                    results = self.model(image)
                    result_image = results[0].plot()
                    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

                    # 将 OpenCV 图像转换为 QImage
                    height, width, channel = result_image.shape
                    bytes_per_line = 3 * width
                    q_img = QImage(result_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(q_img)

                    # 在界面上显示图像
                    self.image_label.setPixmap(pixmap)
            except Exception as e:
                print(f"发生错误: {e}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = YoloDetectionApp()
    sys.exit(app.exec_())
