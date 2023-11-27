#print("################################## line 0 #########################")
import sys
import cv2 
import imutils
from PIL import Image
import numpy as np
import torch
from yoloDet import YoloTRT
from yoloDet_vehicles import YoloTRT_vehicles
from PID_segmentation import PID_Seg, input_transform, load_pretrained
from torch2trt import TRTModule
import pycuda.autoinit  # Problem
import pycuda.driver as cuda

from Config import is_intersecting, is_close_camVlight, is_car_colsing, is_close_camVscross,ROI, Cam_man_pose
from Config import WHEAT1, SPRINGGREEN, RED, GREEN,BLACK, BLUE, YELLOW
from Config import ang, plot_one_box
import time
import tensorrt as trt
import engine as eng
import inference as inf

from fdm_opti import feature_distribution_matching
#import concurrent.futures

from camera.myassis import engine_speak

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)

HEIGHT = 256
WIDTH = 512

SIZE =  (WIDTH,HEIGHT)

from csi_camera import CSI_Camera

show_fps = False

def perform_inference(model, frame):
    return model.Inference(frame)

# Simple draw label on an image; in our case, the video frame
def draw_label(cv_image, label_text, label_position):
	font_face = cv2.FONT_HERSHEY_SIMPLEX
	scale = 0.5
	color = (255,255,255)
	thickness = cv2.FILLED
	# You can get the size of the string with cv2.getTextSize here
	cv2.putText(cv_image, label_text, label_position, font_face, scale, color, 1, cv2.LINE_AA)

# Read a frame from the camera, and draw the FPS on the image if desired
# Return an image
def read_camera(csi_camera,display_fps):
	_ , camera_image=csi_camera.read()
	if display_fps:
		draw_label(camera_image, "Frames Displayed (PS): "+str(csi_camera.last_frames_displayed),(10,20))
		draw_label(camera_image, "Frames Read (PS): "+str(csi_camera.last_frames_read),(10,40))
	return camera_image

# Good for 1280x720
DISPLAY_WIDTH=512
DISPLAY_HEIGHT=256
# For 1920x1080
# DISPLAY_WIDTH=960
# DISPLAY_HEIGHT=540

#3264 x 2464 FR = 21 fps
SENSOR_MODE_2464=0
#3264 x 2464 FR = 28 fps
SENSOR_MODE_2464_2=1
# 1920x1080, 30 fps
SENSOR_MODE_1080=2
# 1640x1232, 30 fps
SENSOR_MODE_1232=3
# 1280x720, 60 fps
SENSOR_MODE_720=4
# 1280x720, 120 fps
SENSOR_MODE_720_2=5

def main():

	# use path for library and engine file
	model = YoloTRT(library="JetsonYolov5/yolov5/build/libmyplugins.so", engine="JetsonYolov5/yolov5/build/best.engine", conf=0.5, yolo_ver="v5")
	#print("################################## line 3 #########################")
	model_vehicle = YoloTRT_vehicles(library="JetsonYolov5_vehicles/yolov5/build/libmyplugins.so", engine="JetsonYolov5_vehicles/yolov5/build/best_vehicle_detection.engine", conf=0.7, yolo_ver="v5")


	serialized_plan_fp32 = "Model/PIDNet_S_Cityscapes_test__.plan"
	engine = eng.load_engine(trt_runtime, serialized_plan_fp32)
	h_input, d_input, h_output, d_output, stream = inf.allocate_buffers(engine, 1, trt.float32)

	W = 512
	H = 256

	P1_ROI, P2_ROI =ROI(W,H)
	P1_Cam, P2_Cam = Cam_man_pose(W,H)

	rectangle_camera_man = [P1_Cam[0], P1_Cam[1] ,P2_Cam[0], P2_Cam[1]]
	rectangle_roi = [P1_ROI[0], P1_ROI[1] ,P2_ROI[0], P2_ROI[1]]

	point_limit_left = (W//5, H//3)
	point_limit_RIGHT = (4*W//5, H//3)

	commun = (W//2, H)
	point_a = (512, 256)
	point_b = point_limit_RIGHT
	point_c = point_limit_left

	lineA = (commun,point_a) #horiz
	lineB = (commun,point_b) #white
	lineC = (commun,point_c)
	alpha = ang(lineA, lineB)
	beta = ang(lineA, lineC)

	#out = cv2.VideoWriter('results/output_rome2***.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, (W, H))
	# Process every 10th frame of the input video
	subsampling_rate = 5
	frame_count = 0
	total_process_time = 0
	pose = (W//2, 20)
	vehicle_warning = False
	#print("################################## line 6 #########################")

	image_reference = cv2.imread('utils2/frankfurt_000000_002196_leftImg8bit.png')

	image_reference_ = cv2.resize(image_reference, (512,256))
	_, _ = model.Inference(image_reference_)
	_, _ = model_vehicle.Inference(image_reference_)
	_ = PID_Seg(image_reference_,image_reference_,engine, h_input, d_input, h_output, d_output, stream, HEIGHT, WIDTH)[1]

	print('#########################################')

	print('OPEN THE CAMERA')

	print('#########################################')



	left_camera = CSI_Camera()
	left_camera.create_gstreamer_pipeline(
			sensor_id=0,
			sensor_mode=SENSOR_MODE_1232,
			framerate=30,
			flip_method=2,
			display_height=DISPLAY_HEIGHT,
			display_width=DISPLAY_WIDTH,
	)
	left_camera.open(left_camera.gstreamer_pipeline)
	left_camera.start()
	cv2.namedWindow("Navigation System", cv2.WINDOW_AUTOSIZE)

	if (
		not left_camera.video_capture.isOpened()
	 ):
		# Cameras did not open, or no camera attached

		print("Unable to open any cameras")
		# TODO: Proper Cleanup
		SystemExit(0)
	try:
		# Start counting the number of frames read and displayed
		left_camera.start_counting_fps()
		while cv2.getWindowProperty("Navigation System", 0) >= 0 : # after this

			frame=read_camera(left_camera,False)
			#print('img size', img.shape)
			#frame = cv2.resize(img, (512,256))
			vehicle_warning = False
			pose = (W//2, 20)
			seg_frame = frame.copy()
			seg_frame = feature_distribution_matching(seg_frame,image_reference,'x.png')
			process_start_time = time.time()

			if frame_count % subsampling_rate == 0:

				detections, t = model.Inference(frame)
				detections_vehicles, t2 = model_vehicle.Inference(frame)

				
				# Initialize some variables
				crossing_lines_detected = False
				traffic_light_detected = False
				traffic_light_color = None

				for p in detections_vehicles:
					class_name = p['class']
					x1, y1, x2, y2 =  [int(i) for i in p['box']]
					p1, p2 = (x1, y1), (x2, y2)  
					plot_one_box(p1,p2, frame, label = class_name, color = [232,15,15], line_thickness = 1 )    
					confidence = p['conf']
					if(confidence > 0.7):
						car_center_x = (x1 + x2) // 2
						car_center_y = (y1+y2) // 2
						#cv2.line(frame, (frame.shape[1]//2, frame.shape[0]), (car_center_x,car_center_y), (255, 255,255), 2)
						angle = ang(lineA, (commun,(car_center_x,car_center_y)))

						if(angle > alpha and angle < beta):
							vehicle_warning = True
							if (angle > 85 and angle < 105):
								msg = "Vehicle is in front"
							elif (angle - 105 >0):
								msg = "Vehicle is in your left"
							elif (angle <85):
								msg ="Vehicle is in your right"
							
							
				'''for obj in detections:
					print(obj['class'], obj['conf'], obj['box'])
					print("FPS: {} sec".format(1/t))'''

				#print("################################## line 11 #########################")
				
				for p in detections:
					class_name = p['class']
					x1, y1, x2, y2 =  [int(i) for i in p['box']]
					p1, p2 = (x1, y1), (x2, y2)         
					#cv2.rectangle(frame, p1,p2, (241,100,248), 2)
					if (class_name =='red light') :
						color = RED
					elif (class_name =='green light') :
						color = GREEN
					else :
						color = (145,30,162)
					plot_one_box(p1,p2, frame, label = class_name, color = color, line_thickness = 1 )      
					#cv2.putText(frame, class_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
					confidence = p['conf']

					if class_name == 'crossline' and confidence > 0.6:
						# Camera man is close to the crossing lines
						crossing_lines_detected = True
						#x1, y1, x2, y2 = [int(i) for i in p['box']]
						#pc1, pc2 = (x1, y1), (x2, y2)
						rectangle_cross = [p1[0], p1[1] ,p2[0], p2[1]]

						

					elif class_name in ['red light', 'green light'] and confidence > 0.5:
						# Traffic light detected
						traffic_light_detected = True
						traffic_light_color = class_name
						#x1, y1, x2, y2 = [int(i) for i in p['box']]                
						#pl1, pl2 = (x1, y1), (x2, y2)
						rectangle_light = [p1[0], p1[1] ,p2[0], p2[1]]
				#print("################################## line 12 #########################")

				order = PID_Seg(seg_frame,frame,engine, h_input, d_input, h_output, d_output, stream, HEIGHT, WIDTH)[1]
				#print("################################## line 13* #########################")


				if crossing_lines_detected:
					#print("######### Detected crossing lines #######")
					if is_close_camVscross(frame,rectangle_cross,rectangle_camera_man,30):
						#print("######### crossing line is close #######")
						#cv2.putText(frame, 'Close To The CrossLine', (W//2, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, RED, 2)
						if(traffic_light_detected):
							#print("######### traffic_light_detected #######")
							if traffic_light_color == 'red light':
								if(is_close_camVlight(rectangle_light,rectangle_camera_man,75)):
									#print("######### On crossline and light is red #######")
									order = 'Keep Going'
								else:
									#print("######### traffic_light is red #######")
									order = 'Stop'
							else:
								#print("######### traffic_light is green #######")
								order = 'Go Forward' 
						else :
							order = 'Crosswalk, But no traffic lights detected'
					else :
						pose = (W//2-170, 50)
						center_x_cross = (rectangle_cross[0]+rectangle_cross[2])//2
						if( center_x_cross < W//3):
							semi_order = 'left'
						elif (center_x_cross >= W//3 and center_x_cross < 2*W//3):
							semi_order = 'forward'
						else :
							semi_order = 'right'
						order = 'Go ' + semi_order + ', there is a crosswalk' #+ PID_Seg(seg_frame,engine, h_input, d_input, h_output, d_output, stream, HEIGHT, WIDTH)[1]
				else:
					#print("######### NO Detected crossing lines #######")

					order = PID_Seg(seg_frame,frame,engine, h_input, d_input, h_output, d_output, stream, HEIGHT, WIDTH)[1]
					#print("######### Order is  ",order, " #######")

				#print("################################## line 13 #########################")
				#cv2.putText(frame, order, pose , cv2.FONT_HERSHEY_SIMPLEX, 0.75, RED, 2)
				cv2.putText(frame, order, pose, 0, 2 / 3, [10, 10, 255], thickness=2, lineType=cv2.LINE_AA)

				#engine_speak(order)
				if vehicle_warning ==True:
					pose_warning = (W//2-100, H-15)
					#cv2.putText(frame, msg, pose_warning, cv2.FONT_HERSHEY_SIMPLEX, 0.75, RED, 2)
					cv2.putText(frame, msg, pose_warning, 0, 2 / 3, [200, 0, 255], thickness=2, lineType=cv2.LINE_AA)
					#engine_speak(msg)

			process_time = time.time() - process_start_time
			total_process_time += process_time
			frame_count += 1

			if show_fps:
				draw_label(frame, "Frames Displayed (PS): "+str(left_camera.last_frames_displayed),(10,20))
				draw_label(frame, "Frames Read (PS): "+str(left_camera.last_frames_read),(10,40))
			cv2.imshow("Navigation System", frame)
			#out.write(frame) # write the frame to the output video
			left_camera.frames_displayed += 1
			keyCode = cv2.waitKey(5) & 0xFF
			# Stop the program on the ESC key
			if keyCode == 27:
				break
		average_latency = total_process_time/frame_count
		print('Avergae latency', average_latency)
	finally:
		left_camera.stop()
		left_camera.release()
		#out.release() 
		cv2.destroyAllWindows()


if __name__ == "__main__":
	main()