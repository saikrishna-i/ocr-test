from paddleocr import PaddleOCR,draw_ocr
# Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# You can set the parameter `lang` as `ch`, `en`, `fr`, `german`, `korean`, `japan`
# to switch the language model in order.
"""
dowload links:
    det_model = https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar
    rec_model = https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar
    cls_path  = https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar

extract the .tar files and place the folder in downloads
"""


#ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory
ocr = PaddleOCR(det_model_dir='downloads/en_PP-OCRv3_det_infer', rec_model_dir='downloads/en_PP-OCRv4_rec_infer', cls_model_dir='downloads/ch_ppocr_mobile_v2.0_cls_infer',use_angle_cls=True, lang='en')
img_path = 'downloads/254.jpg'
result = ocr.ocr(img_path, cls=True)
for idx in range(len(result)):
    res = result[idx]
    for line in res:
        print(line)


# draw result
from PIL import Image
result = result[0]
image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
im_show = draw_ocr(image, boxes, txts, scores,font_path='downloads/simfang.ttf')
im_show = Image.fromarray(im_show)
im_show.save('downloads/254_result.jpg')
