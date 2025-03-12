import cv2
import pytesseract
import re

img = cv2.imread('../../img/image3.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 將彩色圖片轉換為灰度圖片

blur = cv2.GaussianBlur(gray, (11, 11), 0) # 對灰度圖片進行高斯模糊，以減少雜訊

edge = cv2.Canny(blur, 30, 150) # 使用 Canny 邊緣檢測演算法，從模糊後的圖片中找出邊緣

cv2.imshow('edge', edge)
cv2.waitKey(0)

contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 在邊緣影像中尋找輪廓
contours = sorted(contours, key=cv2.contourArea, reverse=True)[: 10] # 取出前 10 個輪廓 (大到小排序)

for item in contours:
    rect = cv2.boundingRect(item) # 計算輪廓的最小外接矩形
    x = rect[0] # 外接矩形左上角 x 座標
    y = rect[1] # 外接矩形左上角 y 座標
    weight = rect[2] # 外接矩形寬度
    height = rect[3] # 外接矩形高度
    if weight / height > 4: # 判斷外接矩形的寬高比是否大於 4，過濾掉可能不是車牌的區域
        cv2.rectangle(img, (x, y), (x + weight, y + height), (0, 0, 255), 3) # 在原始圖片上，用紅色矩形框標示可能的車牌區域
        cropped_plate = gray[y: y + height, x: x + weight] # 從灰度圖片中，根據外接矩形裁切出可能的車牌區域
        cv2.imshow('cropped_plate', cropped_plate)
        cv2.waitKey(0)

        text = pytesseract.image_to_string(cropped_plate, lang='eng') # 使用 pytesseract OCR 辨識裁切出的車牌區域文字，語言設定為英文
        text = text.replace(" ", "") # 移除辨識結果中的空格
        ans = re.search(r'[A-z]{2}-[0-9]{3}-[A-Z]{2}', text).group() # 使用正規表示式，從辨識文字中搜尋符合車牌格式 (兩英文字母-三數字-兩英文字母) 的字串
        print("車牌:", ans)

cv2.destroyAllWindows()