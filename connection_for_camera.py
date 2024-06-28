
import cv2
from threading import Thread, Lock
import numpy as np
import json
from Drone import Drone, CurrentsDrone
from Order import Circle, Rectangle, Any
from PIL import Image


def on_Click(event, x, y, flag, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if Camera.count < Camera.amount_points:
            Camera.image_point.append([x, y])
            Camera.count += 1
            print(Camera.image_point)
            print(Camera.count)
        else:
            pass

class Camera:
    # Счетчик фотографик
    count = 0

    # Координаты проекций в 2d точках
    image_point = []

    # Основная характеристика матрицы
    camera_matrix = []

    # Основные характеристики камеры для метода cv2.solvePnP()
    rvec, tvec = [], []

    rvecs, tvesk = [], []

    # Основная матрица искажения
    dist = []

    # Координаты 3d очек в мировых координатах
    position_point = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [2, 0, 0],
        [0, 1, 0],
        [0, 2, 0],
        [1, 1, 0]], dtype=np.float32)

    objPoint = []

    imgPoint = []

    callback_count = True

    amount_points = position_point.shape[0]

    file = "data_for_camera.json"

    data = {}

    def __init__(self, ip_port='169.254.125.170', port=554, username='drom', password='DRom2022'):
        self.ip_port = ip_port
        self.port = port
        self.username = username
        self.password = password
        self.url = f'rtsp://{username}:{password}@{ip_port}:{port}/h264Preview_01_main'
        self.revelant = r'C:\Users\user\Desktop\Проекты\sam\Image Calibration'

        #self.rvec = []
        #self.tvec = []
        #self.tvecs = []
        #self.rvecs = []

        self.TRANSLATION = False
        self.READ_ALL_PARAMS = True
        self.HAVE_VALUE_FOR_MATRIX = False

    def start_translation(self, queue):
        # reading data for camera
        if self.HAVE_VALUE_FOR_MATRIX:
            self.read_params_for_camera()

        lock = Lock()

        # Get data about drone
        example_drone: Drone = queue.get()

        info_circle = example_drone.circle
        info_rectangle = example_drone.rectangle
        info_any = example_drone.any

        all_drones: list[Drone] = []
        array_position = []

        # main start
        cap = cv2.VideoCapture(0)

        while True:
            array_position.clear()
            ret, frame = cap.read()

            if not ret:
                break

            while not queue.empty():
                with lock:
                    all_drones = queue.get()

            try:
                for drone in all_drones:
                    array_position.append(np.array(drone.position, np.float32))
            except TypeError:
                print("Возникла проблема с типом")
                break

            if not array_position:
                cv2.imshow("Main window", frame)
            else:
                points_center_2d, point_circle = self.get_projection_point(array_position, info_circle.setting.R)

                if info_rectangle.setting.is_active:   # is_active
                    self.draw_box(frame, points_center_2d, info_rectangle)

                if info_circle.setting.is_active:
                    self.draw_circle(point_circle, 2, frame, info_circle.setting.color)

                if info_any.setting.is_active:
                    if info_circle.setting.is_Perspective:
                        position_transform_matrix, depicted_point_will, projection_axes = self.get_perspective_points_for_matrix(array_position, info_any)
                        Matrixs = self.get_transform_matrix(position_transform_matrix, depicted_point_will)
                        img_in_imgs, mask_in_masks = self.do_PerspectiveTransform(info_any.img, info_any.mask, Matrixs)
                        frame = self.add_image_perspective(frame, img_in_imgs, mask_in_masks)

                        print("projection_axes", projection_axes)
                        print("point_center_2d", points_center_2d)
                        self.draw_axis(points_center_2d, projection_axes, frame)
                    else:
                        self.add_image(frame, info_any, points_center_2d)
                    cv2.imshow("Main window", frame)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def check(self):
        if self.HAVE_VALUE_FOR_MATRIX and self.HAVE_VALUE_FOR_MATRIX:
            raise Exception("Не могут значения HAVE_VALUE_FOR_MATRIX и HAVE_VALUE_FOR_MATRIX быть одновременно True")

    def setting_camera(self):
        self.check()

        if self.READ_ALL_PARAMS:
            self.read_all_params_for_camera()

        if self.HAVE_VALUE_FOR_MATRIX:
            self.read_params_for_camera()

        frame_size = (1280, 720)

        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            cv2.imshow("State Setting", frame)

            k = cv2.waitKey(5)
            if k & 0xFF == ord('q'):
                break

            elif (k & 0xFF == ord('w')):
                self.find_corners_for_camera(frame)
                print(f"Я обработал {Camera.count} фотографию удачно")

            elif (k & 0xFF == ord('i')):
                try:
                    self.calibrate_camera(frame_size)
                except Exception as e:
                    print(e)

            elif (k & 0xFF == ord('a')):
                if self.callback_count:
                    print("Активация callback")
                    cv2.setMouseCallback('State Setting', on_Click)
                    self.callback_count = False

            elif k & 0xFF == ord('e'):
                try:
                    Camera.image_point = np.array(Camera.image_point, dtype=np.float32)
                    print(Camera.image_point)
                    self.get_data_PnP()
                except Exception as e:
                    print(e)

            elif k & 0xFF == ord('s'):
                try:
                    point_3d = [[]]
                    # first check:
                    for i in [1, 2, 3, 4, 5, 6]:
                        for j in [1, 2, 3, 4, 5, 6]:
                            point_3d[0].append([i, j, 0])

                    point_3d = np.array(point_3d, np.float32)
                    point_3d = point_3d.reshape(-1, 3)
                    _, point_circle = self.get_projection_point(point_3d, 0.5)
                    self.draw_circle_setting(point_circle, 2, frame)

                except Exception as e:
                    print(e)

            elif (k & 0xFF == ord('r')):
                self.TRANSLATION = True
                break

        cap.release()
        cv2.destroyWindow('State Setting')

    def find_corners_for_camera(self, frame):
        chess_board = (9, 6)

        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        objp = np.zeros((chess_board[0]*chess_board[1], 3), np.float32)

        objp[:, :2] = np.mgrid[0:chess_board[0], 0:chess_board[1]].T.reshape(-1, 2)

        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.waitKey(1000)

        ret, corners = cv2.findChessboardCorners(gray_img, chess_board, None)
        print(ret)

        if ret:
            Camera.count += 1

            Camera.objPoint.append(objp)
            corners_two = cv2.cornerSubPix(gray_img, corners, (11, 11), (-1, -1), criteria)

            Camera.imgPoint.append(corners)
            cv2.drawChessboardCorners(frame, chess_board, corners_two, ret)

            cv2.imshow("State Setting", frame)
            cv2.waitKey(1000)

    def calibrate_camera(self, frame_size):

        Camera.objPoint = np.array(Camera.objPoint, dtype=np.float32)
        Camera.imgPoint = np.array(Camera.imgPoint, dtype=np.float32)

        ret, camera_matrix, dist, rvecs, tvesk = cv2.calibrateCamera(Camera.objPoint, Camera.imgPoint, frame_size, None, None)

        #camera_matrix
        camera_matrix = np.round(camera_matrix, 2)

        Camera.camera_matrix = camera_matrix
        camera_matrix = camera_matrix.tolist()

        #dist
        dist = np.round(dist)

        Camera.dist = dist
        dist = dist.tolist()

        #rvec
        rvecs = [np.round(arr, 2) for arr in rvecs]
        Camera.rvecs = rvecs

        rvecs = [arr.tolist() for arr in rvecs]

        #tvesk
        tvesk = [np.round(arr, 2) for arr in tvesk]
        Camera.tvesk = tvesk

        tvesk= [arr.tolist() for arr in tvesk]


        #Add value for dictionary
        Camera.data["camera_matrix"] = camera_matrix
        Camera.data["distirtion"] = dist
        Camera.data["rotation_vector"] = rvecs
        Camera.data["translation_vector"] = tvesk

        with open(Camera.file, "w", encoding='utf-8') as writter:
            json.dump(Camera.data, writter)

        print("Данные сохранены")

    def read_params_for_camera(self):
        file = "data_for_camera.json"
        with (open(file, 'r', encoding='utf-8') as reader):
            data = json.load(reader)
            try:
                Camera.camera_matrix = data['camera_matrix']
                Camera.camera_matrix = np.array(Camera.camera_matrix, dtype= np.float32)

                Camera.dist = data['distirtion']
                Camera.dist = np.array(Camera.dist, dtype=np.float32)

                Camera.rvec = data['rotation_vector']
                Camera.rvec = np.array(Camera.rvec, dtype=np.float32)

                Camera.tvec = data['translation_vector']
                Camera.tvec = np.array(Camera.tvec, dtype=np.float32)

            except Exception as e:
                print(e)
    def get_data_PnP(self):
        retval, Camera.rvec, Camera.tvec = cv2.solvePnP(Camera.position_point, Camera.image_point, Camera.camera_matrix, Camera.dist)

        Camera.rvec = np.round(Camera.rvec, 2)
        Camera.tvec = np.round(Camera.tvec, 2)

        with open(Camera.file, 'r') as file:
            Camera.data = json.load(file)
            Camera.data['rvec'] = Camera.rvec.tolist()
            Camera.data['tvec'] = Camera.tvec.tolist()

        with open(Camera.file, "w", encoding='utf-8') as writter:
            json.dump(Camera.data, writter)

    def get_projection_point(self, points_3d, r):
        array_point_circle = self.image_circle(points_3d, r)
        points_center = self.compute_projection_point(points_3d)
        point_2d_array = self.compute_projection_point(array_point_circle)
        return points_center, point_2d_array

    def compute_projection_point(self, points):
        arr_projection = []
        for point_3d in points:
            point_center, _ = cv2.projectPoints(point_3d, Camera.rvec, Camera.tvec, Camera.camera_matrix, Camera.dist)
            point_center_2d = np.ravel(point_center)
            x = int(point_center_2d[0])
            y = int(point_center_2d[1])
            arr_projection.append((x, y))

        return arr_projection

    def draw_circle(self, point_2d_array, r, frame, info_circle):
        for item in point_2d_array:
            cv2.circle(frame, item, r, info_circle, 2)

    def read_all_params_for_camera(self):
        with open(Camera.file, 'r') as file:
            data = json.load(file)
            try:
                Camera.camera_matrix = data['camera_matrix']
                Camera.camera_matrix = np.array(Camera.camera_matrix, dtype=np.float32)

                Camera.dist = data['distirtion']
                Camera.dist = np.array(Camera.dist, dtype=np.float32)

                Camera.rvec = data['rvec']
                Camera.rvec = np.array(Camera.rvec, dtype=np.float32)

                Camera.tvec = data['tvec']
                Camera.tvec = np.array(Camera.tvec, dtype=np.float32)

                Camera.rvecs = data['rotation_vector']
                Camera.rvecs = np.array(Camera.rvecs, dtype=np.float32)

                Camera.tvesk = data['translation_vector']
                Camera.tvesk = np.array(Camera.tvesk, dtype=np.float32)

            except Exception as e:
                print(e)

            finally:
                print(Camera.camera_matrix, type(Camera.camera_matrix), "\n",
                      Camera.rvec, type(Camera.rvec), "\n",
                      Camera.tvec, type(Camera.tvec)," \n",
                      Camera.rvecs, type(Camera.rvecs), "\n",
                      Camera.tvesk, type(Camera.tvesk), "\n",
                      Camera.dist,  type(Camera.dist), "\n"
                      )
                print("Камера готова к использованию")

    def image_circle(self, arr, r):
        arr_ = []
        tethas = np.linspace(0, 2*np.pi, 36)
        for ar in arr:
            for tetha in tethas:
                _x = ar[0] + r * np.cos(tetha)
                _y = ar[1] + r * np.sin(tetha)
                arr_.append(np.array((_x, _y, ar[2]), dtype=np.float32))
        return arr_

    def draw_circle_setting (self, point_2d_array, r, frame):
        for item in point_2d_array:
            cv2.circle(frame, item, r, (255, 255, 255), 2)
        cv2.imshow("State Setting", frame)
        cv2.waitKey(10)

    def draw_box(self, frame, points, info_rectangle):
        for point in points:
            cv2.rectangle(frame, (int(point[0]-info_rectangle.setting.dx), int(point[1] - 30 - info_rectangle.setting.dy)),
            (int(point[0]+info_rectangle.dx_h), int(point[1] - 30 + info_rectangle.setting.dy)), info_rectangle.setting.color, -1)

    def add_image(self, frame, img, points_center_2d):
        im = img.get_cv()
        for point_2d in points_center_2d:
            dx_st, dx_ed, dy_st, dy_ed = self.compute_point_2d_without_perspective(point_2d, im)
            sLice = frame[dx_st:dx_ed, dy_st: dy_ed]
            try:
                if dy_ed < frame.shape[1] and dx_ed < frame.shape[0] and dx_st > 0 and dy_st > 0:
                    masked_image2 = cv2.bitwise_and(im, im, mask=img.mask)
                    masked_image1 = cv2.bitwise_and(sLice, sLice, mask=img.mask)
                    sLice = cv2.addWeighted(masked_image1, 1.0, masked_image2, 0.3, 0)
                    frame[dx_st:dx_ed, dy_st: dy_ed] = sLice
            except Exception as e:
                print(e)

    def add_image_perspective(self, frame, arr_image_transform, mask_in_masks):
        for i, image in enumerate(arr_image_transform):
            mask_in_masks[i] = cv2.bitwise_not(mask_in_masks[i])

            masked_image2 = cv2.bitwise_and(image, image, mask=mask_in_masks[i])
            frame = cv2.addWeighted(frame, 1.0, masked_image2, 0.3, 0)

        return frame

    def compute_point_2d_without_perspective(self, point_2d, im):
            dx_st = point_2d[0] - int(im.shape[0] / 2)
            dx_ed = point_2d[0] + int(im.shape[0] / 2)

            dy_st = point_2d[1] - int(im.shape[1] / 2)
            dy_ed = point_2d[1] + int(im.shape[1] / 2)

            return dy_st, dy_ed, dx_st, dx_ed,

    def get_perspective_points_for_matrix(self, array_position, info_any):
        four_point_for_object = []
        progections_point_arr = []
        projection_axes = []
        for position in array_position:
            far_left = [position[0] - info_any.settings.dx_real, position[1] - info_any.settings.dy_real, position[2]]
            near_left =[position[0] + info_any.settings.dx_real, position[1] - info_any.settings.dy_real, position[2]]
            far_right =[position[0] - info_any.settings.dx_real, position[1] + info_any.settings.dy_real, position[2]]
            near_right =[position[0] + info_any.settings.dx_real, position[1] + info_any.settings.dy_real, position[2]]

            one_body = [far_left, near_left, far_right, near_right]

            four_point_for_object.append(one_body)

            point_axes = []
            point_axes_z = np.array([position[0] , position[1], position[2] + 2], np.float32 )#
            point_axes.append(point_axes_z)#
            point_axes_x =  np.array([position[0] - 2, position[1], position[2]], np.float32)#
            point_axes.append(point_axes_x)#
            point_axes_y =  np.array([position[0] , position[1] - 2, position[2]], np.float32) #
            point_axes.append(point_axes_y)#

            projection_axes.append(self.compute_projection_point(point_axes))#


        progections_point, depicted_point_will = self.compute_projection_point_expand(four_point_for_object)
        progections_point_arr.append(progections_point)
        return progections_point_arr, depicted_point_will, projection_axes

    def compute_projection_point_expand(self, point_drones_perspective):
        perspective_array = []
        depicted_point_will = []
        for i, drone_4p in enumerate(point_drones_perspective):
            drone_4p = np.array(drone_4p, np.float32)
            result = self.compute_projection_point(drone_4p)

            perspective_array.append(result)
            depicted_point_will.append(i)
        return perspective_array, depicted_point_will

    def get_transform_matrix(self, position_transform_matrix, depicted_point):
        Matrix = []

        for i in depicted_point:

            point_4 = [80, 80]
            point_2 = [0, 80]
            point_3 = [80, 0]
            point_1 = [0, 0]

            src_points = np.array([point_1,
                                   point_2,
                                   point_3,
                                   point_4], np.float32)

            dst_points = np.array(position_transform_matrix[0], np.float32)

            if len(dst_points) > 0:
                try:
                    M = cv2.getPerspectiveTransform(src_points, dst_points[i])
                    Matrix.append(M)
                except IndexError as e:
                    print(e)

        return Matrix

    def do_PerspectiveTransform(self, im, mask, M):
        arr_image_transform = []
        mask_image_transform = []

        for i, matrix in enumerate(M):
            output_image = cv2.warpPerspective(im, matrix, (640, 480))
            mask_image = cv2.warpPerspective(mask, matrix, (640, 480))

            arr_image_transform.append(output_image)
            mask_image_transform.append(mask_image)

        return arr_image_transform, mask_image_transform

    def draw_axis(self, point_center_2d, projection_axes, frame):
        for i, body in enumerate(point_center_2d):
            for b in [0, 1, 2]:
                cv2.line(frame, body, projection_axes[i][b], (127, 126, 23), 4)

class Position:
    __slots__ = ('x', 'y', 'z', 't')

    def __init__(self):
        self.x = -2
        self.y = -2
        self.z = 3
        self.t = 0

    def increase_value(self):
        self.x = self.x + np.sin(self.t)
        self.y = self.y - np.cos(self.t)
        self.z = self.y + 2 * np.cos(self.t)
        self.t += np.pi*2/24
        if self.t >= 8*np.pi*2:
            self.t = 0
        return [np.array([self.x, self.y, self.z], np.float32), np.array([self.x-3, self.y-3, self.z], np.float32)]






