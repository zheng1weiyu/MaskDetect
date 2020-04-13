import cv2
import time
import numpy as np
import os
from deploy import classify
from lib.CenterFace.centerface import CenterFace

testImgPath = './res/images/'
saveImgPath='res/Result/'
if not os.path.exists(saveImgPath):
    os.mkdir(saveImgPath)
testVideo = r"res/5.mp4"
saveVideo = r'./res/Result/sdb5-0.58.mp4'



def draw(bboxs, img=None, thresh=0.5, max_size=0):
    img_cp = img.copy()
    len_line = int(img_cp.shape[1] / 5)
    pad_percent = int(img_cp.shape[1] / 2)
    x = int(img_cp.shape[1] / 25)
    y = int(img_cp.shape[0] / 25)
    pad_x = int(img_cp.shape[1] / 50)
    pad_y = int(img_cp.shape[0] / 25)
    pad_text = 5
    font_scale = (img_cp.shape[0] * img_cp.shape[1]) / (750 * 750)
    font_scale = max(font_scale, 0.25)
    font_scale = min(font_scale, 0.75)

    font_thickness = 1
    if max(img_cp.shape[0], img_cp.shape[1]) > 750: font_thickness = 2
    if bboxs.shape[0] == 0: return img
    bboxs = bboxs[np.where(bboxs[:, -1] > thresh)[0]]
    bboxs = bboxs.astype(int)

    cnt_mask = 0
    cnt_nomask = 0
    for bbox in bboxs:
        img_bbox = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        if img_bbox.shape[0] * img_bbox.shape[1] < max_size:
            continue
        (ftype, prob) = classify(img_arr=img_bbox)
        # print('ftype,prob',(ftype,prob))
        prob_font_scale = (img_bbox.shape[0] * img_bbox.shape[1]) / (100 * 100)
        prob_font_scale = max(prob_font_scale, 0.25)
        prob_font_scale = min(prob_font_scale, 0.75)
        # cv2.putText(img_cp, '{0:.2f}'.format(prob), (bbox[0] + 7, bbox[1] - 3),
        #             cv2.FONT_HERSHEY_SIMPLEX, prob_font_scale, (0, 0, 255), 1, lineType=cv2.LINE_AA)
        if ftype == 0:
            cnt_mask += 1
        else:
            cnt_nomask += 1
        color = (0, 0, 255) if ftype else (0, 255, 0)
        cv2.rectangle(img_cp, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

    # cv2.line(img_cp, (x, y), (x + len_line, y), (0, 255, 0), 2)
    # cv2.putText(img_cp, 'Mask', (x + len_line + pad_x, y + pad_text),
    #             cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness, lineType=cv2.LINE_AA)
    #
    # cv2.line(img_cp, (x, y + pad_y), (x + len_line, y + pad_y), (0, 0, 255), 2)
    # cv2.putText(img_cp, 'face', (x + len_line + pad_x, y + pad_y + pad_text),
    #             cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness, lineType=cv2.LINE_AA)

    # mask_percent = (0 if cnt_mask == 0 else (cnt_mask / (cnt_mask + cnt_nomask))) * 100
    # # print('mask_percent',mask_percent)
    # cv2.putText(img_cp, 'Mask percent: {:.0f}%'.format(mask_percent), (x + pad_percent, y + pad_text),
    #             cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness, lineType=cv2.LINE_AA)
    return img_cp

# detect the direction
def detect_direction(centerface, height=360, width=640, video_path=testImgPath, visualize=False):
    # cap = cv2.VideoCapture(video_path)
    testImgPath = video_path
    testImgData = os.listdir(testImgPath)
    numTest = len(testImgData)
    for index in range(numTest):
        fileName = testImgData[index]
        test_img_path = testImgPath + fileName
        frame = cv2.imread(test_img_path)
        if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
            print('fileName', fileName)
            start_time = time.time()
            frame = cv2.resize(frame, (width, height))
            dets, lms = centerface(frame, threshold=0.5)
            frame = draw(dets, img=frame)

            print("FPS: ", 1.0 / (time.time() - start_time))
            cv2.imwrite(saveImgPath + fileName, frame)
            if visualize:
                max_size = 1024

                if max(frame.shape[0], frame.shape[1]) > max_size:
                    scale = max_size / max(frame.shape[0], frame.shape[1])
                    frame = cv2.resize(frame, None, fx=scale, fy=scale)

                cv2.imshow('sbd', frame)
                cv2.waitKey(1)
                # cv2.waitKeyEx()
                # if cv2.waitKey() & 0xFF == ord('q'):
                #     break
        cv2.destroyAllWindows()


def detect_frame(centerface, height=360, width=640, video_path=None, visualize=False):
    frame = video_path
    if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
        start_time = time.time()
        frame = cv2.resize(frame, (width, height))
        dets, lms = centerface(frame, threshold=0.5)
        frame = draw(dets, img=frame)

        if visualize:
            max_size = 1024
            if max(frame.shape[0], frame.shape[1]) > max_size:
                scale = max_size / max(frame.shape[0], frame.shape[1])
                frame = cv2.resize(frame, None, fx=scale, fy=scale)
            cv2.imshow('sbd', frame)
            cv2.waitKey(0)
            # cv2.waitKeyEx()
            # if cv2.waitKey() & 0xFF == ord('q'):
            #     break
    # cv2.destroyAllWindows()
    return frame


def detect_video(centerface, height=360, width=640, video_path=testVideo, visualize=False):
    # Set Dataloader
    vid_path, vid_writer = None, None
    vid_path = testVideo
    cap = cv2.VideoCapture(video_path)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(saveVideo, fourcc, int(fps), (int(w), int(h)))

    while True:  # 循环读取视频帧
        rval, frame = cap.read()
        if frame is None:
            break

        frame = cv2.resize(frame, (width, height))
        dets, lms = centerface(frame, threshold=0.58)
        frame = draw(dets, img=frame)
        frame = cv2.resize(frame, (int(w), int(h)))
        writer.write(frame)

        # if visualize:
        #     max_size = 1024
        #     if max(frame.shape[0], frame.shape[1]) > max_size:
        #         scale = max_size / max(frame.shape[0], frame.shape[1])
        #         frame = cv2.resize(frame, None, fx=scale, fy=scale)
        #     if isinstance(vid_writer,cv2.VideoWriter):
        #         vid_writer.release()
        #
        #     fps=cap.get(cv2.CAP_PROP_FPS)
        #     w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        #     h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #     fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #     vid_writer=cv2.VideoWriter('video.avi',fourcc,fps,(w,h))
        # vid_writer.write(frame)
        # cv2.imshow('sbd', frame)
        # cv2.waitKey(1)
        # cv2.waitKeyEx()
        # if cv2.waitKey() & 0xFF == ord('q'):
        #     break
        # cv2.destroyAllWindows()
    writer.release()


class MaskDetect():
    def __init__(self, im_width, im_height):
        self.model = CenterFace(im_height, im_width)

    def detectFrame(self, frame):
        frame=detect_frame(self.model, im_height, im_width, video_path=frame, visualize=False)
        return frame

    def detectVideo(self,testVideoPath):
        detect_video(self.model,im_height, im_width, video_path=testVideoPath, visualize=False)

    def detectDirection(self,testImgPath):
        detect_direction(self.model,im_height, im_width, video_path=testImgPath, visualize=False)


if __name__ == "__main__":
    im_width = 640
    im_height = 360
    model = MaskDetect(im_width, im_height)
    # Function 1：process the image

    # frame=cv2.imread("res/images/15.jpg")
    # h,w,_ = frame.shape
    # print(h,w)
    # frame_result=model.detectFrame(frame)
    # image=cv2.resize(frame_result,(int(w),int(h)))
    # print(image.shape[:2])
    # cv2.imshow("image",frame_result)
    # cv2.waitKey(0)

    # Function 2：run the video
    # model.detectVideo(testVideo)

    # Function 3：scan the direction
    model.detectDirection(testImgPath)
