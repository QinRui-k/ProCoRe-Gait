import numpy as np
import os
import cv2
import math
import copy
import imageio
import io
from tqdm import tqdm
from PIL import Image
from lib.utils.tools import ensure_dir
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from lib.utils.utils_smpl import *
import ipdb
from matplotlib.patches import ConnectionPatch

background_color = np.array([50, 50, 50]) / 255
background_color_white = np.array([255, 255, 255]) / 255
    
def render_motion(motion_input, size=4):
    B, T, J, C = motion_input.shape
    fig = plt.figure(figsize=(size * T, size * B * 3))
    axes = []
    rotation=[0,90,270]
    for b_i in range(B):
        row = []
        for ro_i, ro in enumerate(rotation):
            for t_i in range(T):
                idx = ((b_i*3)+ro_i) * T + t_i + 1 
                ax = fig.add_subplot((B*3), T, idx, projection='3d')
                row.append(ax)
                ax.w_xaxis.set_pane_color(background_color_white)
                ax.w_yaxis.set_pane_color(background_color_white)
                ax.w_zaxis.set_pane_color(background_color_white)
                # Get rid of the ticks
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_zticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
                ax.set_zlim(-1, 1)
                ax.w_xaxis.line.set_lw(0.)
                ax.w_yaxis.line.set_lw(0.)
                ax.w_zaxis.line.set_lw(0.)
                ax.grid(False)
                motion_i = motion_input[b_i, t_i]
                point3d(ax, 0, ro, motion_i, color=background_color, label_name='pred', skeleton=True)
                axes.append(row)
    fig.tight_layout()
    fig_image = fig_to_array(fig)
    plt.close('all')
    return fig_image
    
def render_and_save(motion_input, save_path, keep_imgs=False, fps=25, color="#F96706#FB8D43#FDB381", with_conf=False, draw_face=False):
    ensure_dir(os.path.dirname(save_path))
    motion = copy.deepcopy(motion_input)
    if motion.shape[-1]==2 or motion.shape[-1]==3:
        motion = np.transpose(motion, (1,2,0))   #(T,17,D) -> (17,D,T) 
    if motion.shape[1]==2 or with_conf:
        colors = hex2rgb(color)
        if not with_conf:
            J, D, T = motion.shape
            motion_full = np.ones([J,3,T])
            motion_full[:,:2,:] = motion
        else:
            motion_full = motion
        motion_full[:,:2,:] = pixel2world_vis_motion(motion_full[:,:2,:])
        motion2video(motion_full, save_path=save_path, colors=colors, fps=fps)
    elif motion.shape[0]==6890:
        # motion_world = pixel2world_vis_motion(motion, dim=3)
        motion2video_mesh(motion, save_path=save_path, keep_imgs=keep_imgs, fps=fps, draw_face=draw_face)
    else:
        motion_world = pixel2world_vis_motion(motion, dim=3)
        motion2video_3d(motion_world, save_path=save_path, keep_imgs=keep_imgs, fps=fps)
  
        
def render_and_save_1(motion_input, motion_pred, save_path, anomaly_motion=None, anomaly_pose=None, keep_imgs=False, fps=30, color="#F96706#FB8D43#FDB381", with_conf=False, draw_face=False):
    ensure_dir(os.path.dirname(save_path))
    motion = copy.deepcopy(motion_input)
    motion_p = copy.deepcopy(motion_pred)
    if motion.shape[-1]==2 or motion.shape[-1]==3:
        motion = np.transpose(motion, (1,2,0))   #(T,17,D) -> (17,D,T) 
        motion_p = np.transpose(motion_p, (1,2,0))   #(T,17,D) -> (17,D,T) 
        # set the feet to ground
        min_z = motion[:,2:,:].min()
        motion[:,2:,:] = motion[:,2:,:] - min_z
        min_z_p = motion_p[:,2:,:].min()
        motion_p[:,2:,:] = motion_p[:,2:,:] - min_z_p
        min_y = motion[:,1:2,:].min()
        motion[:,1:2,:] = motion[:,1:2,:] - min_y
        min_y_p = motion_p[:,1:2,:].min()
        motion_p[:,1:2,:] = motion_p[:,1:2,:] - min_y_p
    if motion.shape[1]==2 or with_conf:
        colors = hex2rgb(color)
        if not with_conf:
            J, D, T = motion.shape
            motion_full = np.ones([J,3,T])
            motion_full[:,:2,:] = motion
        else:
            motion_full = motion
        motion_full[:,:2,:] = pixel2world_vis_motion(motion_full[:,:2,:])
        motion2video(motion_full, save_path=save_path, colors=colors, fps=fps)
    elif motion.shape[0]==6890:
        # motion_world = pixel2world_vis_motion(motion, dim=3)
        motion2video_mesh(motion, save_path=save_path, keep_imgs=keep_imgs, fps=fps, draw_face=draw_face)
    else:
        motion_world = pixel2world_vis_motion(motion, dim=3)
        motion_world_p = pixel2world_vis_motion(motion_p, dim=3)
        motion2video_3d_1(motion_world, motion_world_p, save_path, anomaly_motion=anomaly_motion, anomaly_pose=anomaly_pose)

def fig_to_array(fig):
    fig.canvas.draw()
    fig_image = np.array(fig.canvas.renderer._renderer)
    return fig_image

def point3d(ax, theta1, theta2, loc, color, label_name=None, skeleton=True):
    skeleton_index_human36 = [[0,1,2,3], [0,4,5,6], [0, 7, 8, 9, 10],
                              [8,14,15,16], [8,11,12,13]]

    # 绘制 3D 散点
    x = loc[:, 0]
    y = loc[:, 1]
    z = loc[:, 2]
    # points
    point_size=25
    point_zorder = 10
    line_zorder=3
    line_width=2
    for index in range(17):
        # if skeleton:
        # 区分左右
        point_color = np.array([70, 70, 70]) / 255
        if index in [1,2,3,14,15,16]:
            point_color = np.array([181, 93, 96]) / 255
        elif index in [4,5,6,11,12,13]:
            point_color = np.array([89, 117, 164]) / 255

        # 区分真值和预测
        if label_name=="gt":
            point_color = np.array([200, 200, 200]) / 255
            point_size = 15
            point_zorder=2
        if skeleton:
            points = ax.scatter(
                xs=x[index],  # x 轴坐标
                ys=z[index],  # y 轴坐标
                zs=y[index],  # z 轴坐标
                zdir='z',  #
                c=point_color,  # color
                s=point_size,  # size
                label=label_name,
                    zorder=point_zorder)
        else:
            
            point_size = 100
            point_zorder=2
            points = ax.scatter(
                xs=x[index],  # x 轴坐标
                ys=z[index],  # y 轴坐标
                zs=y[index],  # z 轴坐标
                zdir='z',  #
                c=point_color,  # color
                s=point_size,  # size
                label=label_name,
                zorder=point_zorder,
                alpha=0.3)
            for nums in range(1,5):
                point_size = point_size/2
                point_zorder=2
                points = ax.scatter(
                    xs=x[index],  # x 轴坐标
                    ys=z[index],  # y 轴坐标
                    zs=y[index],  # z 轴坐标
                    zdir='z',  #
                    c=point_color,  # color
                    s=point_size,  # size
                    label=label_name,
                    zorder=point_zorder,
                    alpha=0.2)

    point_size=25
    point_zorder = 10
    line_zorder=2
    line_width=1
    # 绘制骨骼连线
    for i in range(5):
        skeleton_index = skeleton_index_human36[i]
        part_index_num = len(skeleton_index)
        # 区分gt pred
        # 区分左右
        skeleton_color=np.array([150, 150, 150]) / 255
        if label_name == 'gt':
            skeleton_color = np.array([200, 200, 200]) / 255
            line_zorder=1
            line_width=1

        for j in range(part_index_num - 1):
            id1 = skeleton_index[j]
            id2 = skeleton_index[j + 1]
            ax.plot((x[id1], x[id2]), (z[id1], z[id2]), (y[id1], y[id2]),
                    color=skeleton_color,
                    lw=line_width,
                    zorder=line_zorder)


    x_range = max(abs(min(x)), abs(max(x)))
    y_range = max(abs(min(z)), abs(max(z)))
    # ax.set_xlim(-x_range - 100, x_range + 100)
    # ax.set_ylim(-y_range - 50, y_range + 50)
    # ax.set_zlim(min(y) - 50, max(y) + 50)

    # 调整视角
    ax.view_init(
        elev=theta1,  # 仰角
        azim=theta2  # 方位角
    )

     
def pixel2world_vis(pose):
#     pose: (17,2)
    return (pose + [1, 1]) * 512 / 2

def pixel2world_vis_motion(motion, dim=2, is_tensor=False):
#     pose: (17,2,N)
    N = motion.shape[-1]
    if dim==2:
        offset = np.ones([2,N]).astype(np.float32)
    else:
        offset = np.ones([3,N]).astype(np.float32)
        offset[:,:] = 0
    if is_tensor:
        offset = torch.tensor(offset)
    return (motion + offset) * 500 / 2

def vis_data_batch(data_input, data_label, n_render=10, save_path='doodle/vis_train_data/'):
    '''
        data_input: [N,T,17,2/3]
        data_label: [N,T,17,3]
    '''
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True) 
    for i in range(min(len(data_input), n_render)):
        render_and_save(data_input[i][:,:,:2], '%s/input_%d.mp4' % (save_path, i))
        render_and_save(data_label[i], '%s/gt_%d.mp4' % (save_path, i))

def get_img_from_fig(fig, dpi=120):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    return img

def rgb2rgba(color):
    return (color[0], color[1], color[2], 255)

def hex2rgb(hex, number_of_colors=3):
    h = hex
    rgb = []
    for i in range(number_of_colors):
        h = h.lstrip('#')
        hex_color = h[0:6]
        rgb_color = [int(hex_color[i:i+2], 16) for i in (0, 2 ,4)]
        rgb.append(rgb_color)
        h = h[6:]
    return rgb

def joints2image(joints_position, colors, transparency=False, H=1000, W=1000, nr_joints=49, imtype=np.uint8, grayscale=False, bg_color=(255, 255, 255)):
#     joints_position: [17*2]
    nr_joints = joints_position.shape[0]

    if nr_joints == 49: # full joints(49): basic(15) + eyes(2) + toes(2) + hands(30)
        limbSeq = [[0, 1], [1, 2], [1, 5], [1, 8], [2, 3], [3, 4], [5, 6], [6, 7], \
                   [8, 9], [8, 13], [9, 10], [10, 11], [11, 12], [13, 14], [14, 15], [15, 16],
                   ]#[0, 17], [0, 18]] #ignore eyes

        L = rgb2rgba(colors[0]) if transparency else colors[0]
        M = rgb2rgba(colors[1]) if transparency else colors[1]
        R = rgb2rgba(colors[2]) if transparency else colors[2]

        colors_joints = [M, M, L, L, L, R, R,
                  R, M, L, L, L, L, R, R, R,
                  R, R, L] + [L] * 15 + [R] * 15

        colors_limbs = [M, L, R, M, L, L, R,
                  R, L, R, L, L, L, R, R, R,
                  R, R]
    elif nr_joints == 15: # basic joints(15) + (eyes(2))
        limbSeq = [[0, 1], [1, 2], [1, 5], [1, 8], [2, 3], [3, 4], [5, 6], [6, 7],
                   [8, 9], [8, 12], [9, 10], [10, 11], [12, 13], [13, 14]]
                    # [0, 15], [0, 16] two eyes are not drawn

        L = rgb2rgba(colors[0]) if transparency else colors[0]
        M = rgb2rgba(colors[1]) if transparency else colors[1]
        R = rgb2rgba(colors[2]) if transparency else colors[2]

        colors_joints = [M, M, L, L, L, R, R,
                         R, M, L, L, L, R, R, R]

        colors_limbs = [M, L, R, M, L, L, R,
                        R, L, R, L, L, R, R]
    elif nr_joints == 17: # H36M, 0: 'root',
    #                             1: 'rhip',
    #                             2: 'rkne',
    #                             3: 'rank',
    #                             4: 'lhip',
    #                             5: 'lkne',
    #                             6: 'lank',
    #                             7: 'belly',
    #                             8: 'neck',
    #                             9: 'nose',
    #                             10: 'head',
    #                             11: 'lsho',
    #                             12: 'lelb',
    #                             13: 'lwri',
    #                             14: 'rsho',
    #                             15: 'relb',
    #                             16: 'rwri'
        limbSeq = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [8, 11], [8, 14], [9, 10], [11, 12], [12, 13], [14, 15], [15, 16]]

        L = rgb2rgba(colors[0]) if transparency else colors[0]
        M = rgb2rgba(colors[1]) if transparency else colors[1]
        R = rgb2rgba(colors[2]) if transparency else colors[2]

        colors_joints = [M, R, R, R, L, L, L, M, M, M, M, L, L, L, R, R, R]
        colors_limbs = [R, R, R, L, L, L, M, M, M, L, R, M, L, L, R, R]
        
    else:
        raise ValueError("Only support number of joints be 49 or 17 or 15")

    if transparency:
        canvas = np.zeros(shape=(H, W, 4))
    else:
        canvas = np.ones(shape=(H, W, 3)) * np.array(bg_color).reshape([1, 1, 3])
    hips = joints_position[0]
    neck = joints_position[8]
    torso_length = ((hips[1] - neck[1]) ** 2 + (hips[0] - neck[0]) ** 2) ** 0.5
    head_radius = int(torso_length/4.5)
    end_effectors_radius = int(torso_length/15)
    end_effectors_radius = 7
    joints_radius = 7
    for i in range(0, len(colors_joints)):
        if i in (17, 18):
            continue
        elif i > 18:
            radius = 2
        else:
            radius = joints_radius
        if len(joints_position[i])==3:                 # If there is confidence, weigh by confidence
            weight = joints_position[i][2]
            if weight==0:
                continue
        cv2.circle(canvas, (int(joints_position[i][0]),int(joints_position[i][1])), radius, colors_joints[i], thickness=-1)
        
    stickwidth = 2
    for i in range(len(limbSeq)):
        limb = limbSeq[i]
        cur_canvas = canvas.copy()
        point1_index = limb[0]
        point2_index = limb[1]
        point1 = joints_position[point1_index]
        point2 = joints_position[point2_index]
        if len(point1)==3:                             # If there is confidence, weigh by confidence
            limb_weight = min(point1[2], point2[2])
            if limb_weight==0:
                bb = bounding_box(canvas)
                canvas_cropped = canvas[:,bb[2]:bb[3], :]
                continue
        X = [point1[1], point2[1]]
        Y = [point1[0], point2[0]]
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        alpha = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(alpha), 0, 360, 1)
        cv2.fillConvexPoly(cur_canvas, polygon, colors_limbs[i])
        canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
        bb = bounding_box(canvas)
        canvas_cropped = canvas[:,bb[2]:bb[3], :]
    canvas = canvas.astype(imtype)
    canvas_cropped = canvas_cropped.astype(imtype)
    if grayscale:
        if transparency:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_RGBA2GRAY)
            canvas_cropped = cv2.cvtColor(canvas_cropped, cv2.COLOR_RGBA2GRAY)
        else:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY)
            canvas_cropped = cv2.cvtColor(canvas_cropped, cv2.COLOR_RGB2GRAY)
    return [canvas, canvas_cropped]


def motion2video(motion, save_path, colors, h=512, w=512, bg_color=(255, 255, 255), transparency=False, motion_tgt=None, fps=25, save_frame=False, grayscale=False, show_progress=True, as_array=False):
    nr_joints = motion.shape[0]
#     as_array = save_path.endswith(".npy")
    vlen = motion.shape[-1]

    out_array = np.zeros([vlen, h, w, 3]) if as_array else None
    videowriter = None if as_array else imageio.get_writer(save_path, fps=fps)

    if save_frame:
        frames_dir = save_path[:-4] + '-frames'
        ensure_dir(frames_dir)

    iterator = range(vlen)
    if show_progress: iterator = tqdm(iterator)
    for i in iterator:
        [img, img_cropped] = joints2image(motion[:, :, i], colors, transparency=transparency, bg_color=bg_color, H=h, W=w, nr_joints=nr_joints, grayscale=grayscale)
        if motion_tgt is not None:
            [img_tgt, img_tgt_cropped] = joints2image(motion_tgt[:, :, i], colors, transparency=transparency, bg_color=bg_color, H=h, W=w, nr_joints=nr_joints, grayscale=grayscale)
            img_ori = img.copy()
            img = cv2.addWeighted(img_tgt, 0.3, img_ori, 0.7, 0)
            img_cropped = cv2.addWeighted(img_tgt, 0.3, img_ori, 0.7, 0)
            bb = bounding_box(img_cropped)
            img_cropped = img_cropped[:, bb[2]:bb[3], :]
        if save_frame:
            save_image(img_cropped, os.path.join(frames_dir, "%04d.png" % i))
        if as_array: out_array[i] = img
        else: videowriter.append_data(img)

    if not as_array:
        videowriter.close()

    return out_array

def motion2video_3d(motion, save_path, fps=25, keep_imgs = False):
#     motion: (17,3,N)
    videowriter = imageio.get_writer(save_path, fps=fps)
    vlen = motion.shape[-1]
    save_name = save_path.split('.')[0]
    frames = []
    joint_pairs = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [8, 11], [8, 14], [9, 10], [11, 12], [12, 13], [14, 15], [15, 16]]
    joint_pairs_left = [[8, 11], [11, 12], [12, 13], [0, 4], [4, 5], [5, 6]]
    joint_pairs_right = [[8, 14], [14, 15], [15, 16], [0, 1], [1, 2], [2, 3]]
    
    color_mid = "#00457E"
    color_left = "#02315E"
    color_right = "#2F70AF"
    for f in tqdm(range(vlen)):
        j3d = motion[:,:,f]
        fig = plt.figure(0, figsize=(10, 10))
        ax = plt.axes(projection="3d")
        ax.w_xaxis.set_pane_color(background_color_white)
        ax.w_yaxis.set_pane_color(background_color_white)
        ax.w_zaxis.set_pane_color(background_color)
        ax.grid(False)
        ax.set_xlim(-500, 500)
        ax.set_ylim(-500, 500)
        ax.set_zlim(-500, 500)
        ax.view_init(elev=12., azim=80)
        plt.tick_params(left = False, right = False , labelleft = False ,
                        labelbottom = False, bottom = False)
        for i in range(len(joint_pairs)):
            limb = joint_pairs[i]
            xs, ys, zs = [np.array([j3d[limb[0], j], j3d[limb[1], j]]) for j in range(3)]
            if joint_pairs[i] in joint_pairs_left:
                ax.plot(xs, zs, ys, color=color_left, lw=3, marker='o', markerfacecolor='w', markersize=3, markeredgewidth=2) # axis transformation for visualization
            elif joint_pairs[i] in joint_pairs_right:
                ax.plot(xs, zs, ys, color=color_right, lw=3, marker='o', markerfacecolor='w', markersize=3, markeredgewidth=2) # axis transformation for visualization
            else:
                ax.plot(xs, zs, ys, color=color_mid, lw=3, marker='o', markerfacecolor='w', markersize=3, markeredgewidth=2) # axis transformation for visualization
            
        frame_vis = get_img_from_fig(fig)
        videowriter.append_data(frame_vis)
        plt.close()
    videowriter.close()

def motion2video_3d_1(motion_input, motion_pred, save_path, anomaly_motion=None, anomaly_pose=None, keep_imgs=False, fps=30):
#     motion: (17,3,N)
    videowriter = imageio.get_writer(save_path, fps=fps)
    vlen = motion_input.shape[-1]
    save_name = save_path.split('.')[0]
    frames = []
    joint_pairs = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [8, 11], [8, 14], [9, 10], [11, 12], [12, 13], [14, 15], [15, 16]]
    joint_pairs_left = [[8, 11], [11, 12], [12, 13], [0, 4], [4, 5], [5, 6]]
    joint_pairs_right = [[8, 14], [14, 15], [15, 16], [0, 1], [1, 2], [2, 3]]
    
    color_mid = "#00457E"
    color_left = "#02315E"
    color_right = "#2F70AF"
    color_motion_anomal = "#dc8a8c"
    color_pose_anomal = "#f9dcae"
    fig = plt.figure(figsize=(18, 6))
    title_list = ['raw gait', 'generated normal gait', 'anomaly highlight']
    for f in tqdm(range(vlen)):
        for t_i, title in enumerate(title_list):
            if t_i == 1:
                motion = motion_pred
            else:
                motion = motion_input
            ax = fig.add_subplot(1, 3, t_i+1, projection='3d')
            j3d = motion[:,:,f]
            ax.set_title(title)
            ax.w_xaxis.set_pane_color(background_color_white)
            ax.w_yaxis.set_pane_color(background_color_white)
            ax.w_zaxis.set_pane_color(background_color)
            ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.grid(False)
            ax.set_xlim(-500, 500)
            ax.set_ylim(-500, 500)
            ax.set_zlim(0, 800)
            ax.view_init(elev=12., azim=80)
            plt.tick_params(left = False, right = False , labelleft = False ,
                            labelbottom = False, bottom = False)
            for i in range(len(joint_pairs)):
                limb = joint_pairs[i]
                xs, ys, zs = [np.array([j3d[limb[0], j], j3d[limb[1], j]]) for j in range(3)]
                if joint_pairs[i] in joint_pairs_left:
                    ax.plot(xs, zs, ys, color=color_left, lw=3, marker='o', markerfacecolor='w', markersize=3, markeredgewidth=2) # axis transformation for visualization
                elif joint_pairs[i] in joint_pairs_right:
                    ax.plot(xs, zs, ys, color=color_right, lw=3, marker='o', markerfacecolor='w', markersize=3, markeredgewidth=2) # axis transformation for visualization
                else:
                    ax.plot(xs, zs, ys, color=color_mid, lw=3, marker='o', markerfacecolor='w', markersize=3, markeredgewidth=2) # axis transformation for visualization
            
            if t_i==2:
                # 高亮显示异常关节
                if anomaly_motion is not None:
                    anomaly_motion_flag=False
                    for joint_idx in anomaly_motion[f:(f+1)]:
                        if joint_idx>=0:
                            anomaly_motion_flag=True
                            x, z, y = j3d[joint_idx, 0], j3d[joint_idx, 2], j3d[joint_idx, 1]
                            alphas = [0.8,0.6,0.4,0.2]          # 每圈透明度
                            sizes = [10*(3*i+1) for i in range(4)]  # 每圈大小
                            for s, a in zip(sizes, alphas):
                                ax.scatter(x, z, y,
                                        c=color_motion_anomal,
                                        s=s,
                                        alpha=a,
                                        marker='o',
                                        linewidths=0.5 if a == 1.0 else 0)
                    if anomaly_motion_flag:
                        ax.text(
                            -660,500,100,
                            s="motion anomaly",
                            ha='right', va='top',
                            fontsize=12,
                            color=color_motion_anomal, 
                            bbox=dict(
                                boxstyle='round,pad=0.3', # 圆角矩形，内边距
                                facecolor='black',          # 背景框颜色
                                edgecolor='none',         # 背景框边缘颜色
                                alpha=0.3                 # 背景框透明度
                            )
                        )
                if anomaly_pose is not None:
                    anomaly_pose_flag=False
                    for joint_idx in anomaly_pose[f:(f+1)]:
                        if joint_idx>=0:
                            anomaly_pose_flag=True
                            x, z, y = j3d[joint_idx, 0], j3d[joint_idx, 2], j3d[joint_idx, 1]
                            alphas = [0.8,0.6,0.4,0.2]           # 每圈透明度
                            sizes = [10*(3*i+1) for i in range(4)]  # 每圈大小
                            for s, a in zip(sizes, alphas):
                                ax.scatter(x, z, y,
                                        c=color_pose_anomal,
                                        s=s,
                                        alpha=a,
                                        marker='o',
                                        linewidths=0.5 if a == 1.0 else 0)
                    if anomaly_pose_flag:
                        ax.text(
                            -660,500, 0,
                            s="pose anomaly",
                            ha='right', va='top',
                            fontsize=12,
                            color=color_pose_anomal, 
                            bbox=dict(
                                boxstyle='round,pad=0.3', # 圆角矩形，内边距
                                facecolor='black',          # 背景框颜色
                                edgecolor='none',         # 背景框边缘颜色
                                alpha=0.3                 # 背景框透明度
                            )
                        )
        frame_vis = get_img_from_fig(fig)
        videowriter.append_data(frame_vis)
        plt.close()
    videowriter.close()

def plot_pose3d(ax, pose, motion_joints=None, pose_joints=None):
    """
    在给定的 3D 坐标轴上绘制单个 3D 姿态。

    Args:
        ax (Axes3D): Matplotlib 3D 坐标轴对象。
        pose (np.array): 单帧的 3D 姿态数据，形状为 (num_joints, 3)。
        bones (list): 描述关节连接关系的元组列表。
        bone_color (str): 骨骼的颜色。
        joint_color (str): 关节的颜色。
        highlight_joints (list, optional): 需要高亮显示的关节索引列表。
        highlight_colors (list, optional): 对应高亮关节的颜色列表。
    """
    joint_pairs = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [8, 11], [8, 14], [9, 10], [11, 12], [12, 13], [14, 15], [15, 16]]
    joint_pairs_left = [[8, 11], [11, 12], [12, 13], [0, 4], [4, 5], [5, 6]]
    joint_pairs_right = [[8, 14], [14, 15], [15, 16], [0, 1], [1, 2], [2, 3]]
    
    color_mid = "#00457E"
    color_left = "#02315E"
    color_right = "#2F70AF"
    color_motion_anomal = "#dc8a8c"
    color_pose_anomal = "#f9dcae"

    # 绘制关节
    for i in range(len(joint_pairs)):
            limb = joint_pairs[i]
            xs, ys, zs = [np.array([pose[limb[0], j], pose[limb[1], j]]) for j in range(3)]
            if joint_pairs[i] in joint_pairs_left:
                ax.plot(xs, zs, ys, color=color_left, lw=3, marker='o', markerfacecolor='w', markersize=3, markeredgewidth=2) # axis transformation for visualization
            elif joint_pairs[i] in joint_pairs_right:
                ax.plot(xs, zs, ys, color=color_right, lw=3, marker='o', markerfacecolor='w', markersize=3, markeredgewidth=2) # axis transformation for visualization
            else:
                ax.plot(xs, zs, ys, color=color_mid, lw=3, marker='o', markerfacecolor='w', markersize=3, markeredgewidth=2) # axis transformation for visualization
        
    # 高亮显示异常关节
    if motion_joints:
        for joint_idx in motion_joints:
            if joint_idx >=0 :
                x, z, y = pose[joint_idx, 0], pose[joint_idx, 2], pose[joint_idx, 1]
                alphas = [0.8, 0.6, 0.4, 0.2]           # 每圈透明度
                sizes = [10*(3*i+1) for i in range(4)]  # 每圈大小
                for s, a in zip(sizes, alphas):
                    ax.scatter(x, z, y,
                            c=color_motion_anomal,
                            s=s,
                            alpha=a,
                            marker='o',
                            linewidths=0.5 if a == 1.0 else 0)
        
    if pose_joints:
        for joint_idx in pose_joints:
            if joint_idx >=0 :
                x, z, y = pose[joint_idx, 0], pose[joint_idx, 2], pose[joint_idx, 1]
                alphas = [0.8, 0.6, 0.4, 0.2]           # 每圈透明度
                sizes = [10*(3*i+1) for i in range(4)]  # 每圈大小
                for s, a in zip(sizes, alphas):
                    ax.scatter(x, z, y,
                            c=color_pose_anomal,
                            s=s,
                            alpha=a,
                            marker='o',
                            linewidths=0.5 if a == 1.0 else 0)

            
            
def visualize_pose_sequences(sequence1, sequence2, sequence3, abnormal_joints_motion, abnormal_joints_pose, corresponding_frames=None, save_path=None):
    """
    可视化三行 3D 姿态序列。

    Args:
        sequence1 (np.array): 第一行的姿态序列。
        sequence2 (np.array): 第二行的姿态序列。
        sequence3 (np.array): 第三行的姿态序列。
        abnormal_joints_motion (np.array): 运动异常关节index。
        abnormal_joints_pose (np.array): 姿态异常关节index。
        corresponding_frames (list): 包含第二行和第三行对应帧索引的元组列表。
    """
    
    num_frames = max(len(sequence1), len(sequence2), len(sequence3))
    fig = plt.figure(figsize=(num_frames * 2, 6))
    axes_dict = {}  # 存放每个 subplot 的位置
    
    motion_1 = copy.deepcopy(sequence1)
    motion_2 = copy.deepcopy(sequence2)
    motion_p = copy.deepcopy(sequence3)
    if motion_1.shape[-1]==2 or motion_1.shape[-1]==3:
        motion_1 = np.transpose(motion_1, (1,2,0))   #(T,17,D) -> (17,D,T) 
        motion_2 = np.transpose(motion_2, (1,2,0))   #(T,17,D) -> (17,D,T) 
        motion_p = np.transpose(motion_p, (1,2,0))   #(T,17,D) -> (17,D,T) 
    motion_1 = pixel2world_vis_motion(motion_1, dim=3)
    motion_2 = pixel2world_vis_motion(motion_2, dim=3)    
    motion_p = pixel2world_vis_motion(motion_p, dim=3)    
    motion_1 = np.transpose(motion_1, (2,0,1))
    motion_2 = np.transpose(motion_2, (2,0,1))
    motion_p = np.transpose(motion_p, (2,0,1))
    x_min_1 = motion_1[:,:,0].min()
    y_min_1 = motion_1[:,:,1].min()
    z_min_1 = motion_1[:,:,2].min()
    x_min_3 = motion_p[:,:,0].min()
    y_min_3 = motion_p[:,:,1].min()
    z_min_3 = motion_p[:,:,2].min()
    x_min = min(x_min_1, x_min_3)- 10
    y_min = min(y_min_1, y_min_3) 
    z_min = min(z_min_1, z_min_3) 
    x_max_1 = motion_1[:,:,0].max()
    y_max_1 = motion_1[:,:,1].max()
    z_max_1 = motion_1[:,:,2].max()
    x_max_3 = motion_p[:,:,0].max()
    y_max_3 = motion_p[:,:,1].max()
    z_max_3 = motion_p[:,:,2].max()
    x_max = max(x_max_1, x_max_3)+ 10
    y_max = max(y_max_1, y_max_3) 
    z_max = max(z_max_1, z_max_3) 
    for i in range(num_frames):
        # --- 第一行：带异常点高亮的序列 ---
        ax1 = fig.add_subplot(2, num_frames, i + 1, projection='3d')
        axes_dict[(0, i)] = ax1
        ax1.axis('off')
        ax1.view_init(elev=10., azim=80)
        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(z_min, z_max)
        ax1.set_zlim(y_min, y_max)
        # ax1.view_init(elev=12., azim=80)
        pose1 = motion_1[i]
        abnormal_m = abnormal_joints_motion[i:(i+1)]
        abnormal_p = abnormal_joints_pose[i:(i+1)]
        plot_pose3d(ax1, pose1, motion_joints=abnormal_m, pose_joints=abnormal_p)


        # # --- 第二行：序列 A ---
        # ax2 = fig.add_subplot(3, num_frames, i + 1 + num_frames, projection='3d')
        # axes_dict[(1, i)] = ax2
        # ax2.axis('off')
        # ax2.view_init(elev=10., azim=80)
        # # ax2.set_xlim(-500, 500)
        # # ax2.set_ylim(0, 1000)
        # # ax2.set_zlim(-500, 500)
        # # ax2.view_init(elev=12., azim=80)
        # if i < len(sequence2):
        #     pose2 = motion_2[i]
        #     plot_pose3d(ax2, pose2)


        # --- 第三行：序列 B ---
        ax3 = fig.add_subplot(2, num_frames, i + 1 + 1 * num_frames, projection='3d')
        axes_dict[(1, i)] = ax3
        ax3.axis('off')
        ax3.view_init(elev=10., azim=80)
        ax3.set_xlim(x_min, x_max)
        ax3.set_ylim(z_min, z_max)
        ax3.set_zlim(y_min, y_max)
        # ax3.view_init(elev=12., azim=80)
        if i < len(sequence3):
            pose3 = motion_p[i]
            plot_pose3d(ax3, pose3)


    # --- 绘制第二行和第三行之间的对应连线 ---
    if corresponding_frames is not None:
        for frame_idx2, frame_idx3 in corresponding_frames.items():
            # 获取第二行和第三行对应帧的子图坐标轴
            ax2 = axes_dict[(0, frame_idx3)]  # 第 2 行（索引 1）
            ax3 = axes_dict[(1, frame_idx2)]  # 第 3 行（索引 2）

            # 获取这两个子图的坐标系中心位置
            xyA = (0.5, -0.15)     # 下边界中点（第2行）
            xyB = (0.5, 1.15)     # 上边界中点（第3行）

            con = ConnectionPatch(
                xyA=xyA, coordsA=ax2.transAxes,
                xyB=xyB, coordsB=ax3.transAxes,
                arrowstyle='-', linestyle='--', color='gray', linewidth=2
            )
            fig.add_artist(con)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=400)
    plt.close()
    
    
def motion2video_mesh(motion, save_path, fps=25, keep_imgs = False, draw_face=True):
    videowriter = imageio.get_writer(save_path, fps=fps)
    vlen = motion.shape[-1]
    draw_skele = (motion.shape[0]==17)
    save_name = save_path.split('.')[0]
    smpl_faces = get_smpl_faces()
    frames = []
    joint_pairs = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [8, 11], [8, 14], [9, 10], [11, 12], [12, 13], [14, 15], [15, 16]]

    
    X, Y, Z = motion[:, 0], motion[:, 1], motion[:, 2]
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    
    for f in tqdm(range(vlen)):
        j3d = motion[:,:,f]
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        fig = plt.figure(0, figsize=(8, 8))
        ax = plt.axes(projection="3d", proj_type = 'ortho')
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.view_init(elev=-90, azim=-90)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        
        # plt.savefig("filename.png", transparent=True, bbox_inches="tight", pad_inches=0)
        
        if draw_skele:
            for i in range(len(joint_pairs)):
                limb = joint_pairs[i]
                xs, ys, zs = [np.array([j3d[limb[0], j], j3d[limb[1], j]]) for j in range(3)]
                ax.plot(-xs, -zs, -ys, c=[0,0,0], lw=3, marker='o', markerfacecolor='w', markersize=3, markeredgewidth=2) # axis transformation for visualization
        elif draw_face:
            ax.plot_trisurf(j3d[:, 0], j3d[:, 1], triangles=smpl_faces, Z=j3d[:, 2], color=(166/255.0,188/255.0,218/255.0,0.9))
        else:
            ax.scatter(j3d[:, 0], j3d[:, 1], j3d[:, 2], s=3, c='w', edgecolors='grey')
        frame_vis = get_img_from_fig(fig, dpi=128)
        plt.cla()
        videowriter.append_data(frame_vis)
        plt.close()
    videowriter.close()

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def bounding_box(img):
    a = np.where(img != 0)
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    return bbox
