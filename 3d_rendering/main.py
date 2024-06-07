import math

import numpy
import numpy as np
import cv2 as cv
import copy

FOV_X = math.radians(61)
FOV_Y = math.radians(37)
WIDTH = 1280
HEIGHT = 720

FOCAL_LENGTH_X = WIDTH / (2 * math.tan(FOV_X / 2.0))
FOCAL_LENGTH_Y = HEIGHT / (2 * math.tan(FOV_Y / 2.0))

INTRINSIC_MATRIX = np.array([
    [FOCAL_LENGTH_X, 0, WIDTH / 2, 0],
    [0, FOCAL_LENGTH_Y, HEIGHT / 2, 0],
    [0, 0, 1, 0]
])

# affine vectors of shape's corners
SHAPE = np.array([
    [1, 1, -1, -1, -1, -1, 1, 1],
    [1, -1, -1, 1, 1, -1, -1, 1],
    [50, 50, 50, 50, 48, 48, 48, 48],
    [1, 1, 1, 1, 1, 1, 1, 1]
])

WINDOW_NAME = "window"
SPEED = 0.5  # camera speed (units/millisecond)
ANGULAR_SPEED = math.radians(0.8)  # camera angular angle (rad/millisecond)
POINT_COLOR = (1, 255, 255)  # BGR
LINE_COLOR = (255, 255, 0)  # BGR
POINT_RADIUS = 10
LINE_WIDTH = 3
run = True

cam_pos = [0.0, 0.0, 0.0]  # x y z
cam_rot = [0.0, 0.0, 0.0]  # pitch yaw roll


def get_extrinsic_matrix() -> np.array:
    global cam_pos, cam_rot
    transform_pos = np.array([
        [1, 0, 0, -cam_pos[0]],
        [0, 1, 0, -cam_pos[1]],
        [0, 0, 1, -cam_pos[2]],
        [0, 0, 0, 1]
    ])
    transform_rot, _ = cv.Rodrigues(np.array([-cam_rot[0], -cam_rot[1], -cam_rot[2]]))
    transform_rot = np.array([  # turn it to an affine matrix
        [transform_rot[0][0], transform_rot[0][1], transform_rot[0][2], 0],
        [transform_rot[1][0], transform_rot[1][1], transform_rot[1][2], 0],
        [transform_rot[2][0], transform_rot[2][1], transform_rot[2][2], 0],
        [0, 0, 0, 1],
    ])
    return transform_rot @ transform_pos


def get_perspective_proj_matrix() -> np.array:
    return INTRINSIC_MATRIX @ get_extrinsic_matrix()


def handle_input(key: int):
    global cam_pos, cam_rot, run
    if key == ord('w'):
        cam_pos[2] += SPEED
    elif key == ord('s'):
        cam_pos[2] -= SPEED
    elif key == ord('a'):
        cam_pos[0] -= SPEED
    elif key == ord('d'):
        cam_pos[0] += SPEED
    elif key == ord('z'):
        cam_pos[1] -= SPEED
    elif key == ord('c'):
        cam_pos[1] += SPEED
    elif key == ord('q'):
        cam_rot[1] += ANGULAR_SPEED
    elif key == ord('e'):
        cam_rot[1] -= ANGULAR_SPEED
    elif key == 27:  # esc
        run = False


def main():
    blank = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)  # creates an empty black image
    while run:
        img = copy.copy(blank)
        projected_shape = get_perspective_proj_matrix() @ SHAPE
        points_positions = np.transpose(projected_shape)

        # these are the projected positions of the shape's points in the frame
        points_frame = [(int(point[0] / point[2]), int(point[1] / point[2])) for point in points_positions]

        # draw lines
        for i in range(len(points_frame)):
            point = points_frame[i]
            next_point = points_frame[(i + 1) % len(points_frame)]
            cv.line(img, point, next_point, LINE_COLOR, LINE_WIDTH)

        # draw points
        for point in points_frame:
            cv.circle(img,
                      point,
                      int(POINT_RADIUS * 0.5),  # this calculation is to compensate for the lines width
                      POINT_COLOR,
                      POINT_RADIUS  # the width is equal to the radius so the circle will be full
                      )
        cv.imshow(WINDOW_NAME, img)
        key = cv.waitKey(1)
        handle_input(key)


if __name__ == '__main__':
    main()
