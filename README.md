TITLE:

Smart Weather-Adaptive ADAS AI-Based Driving Assistance for Adverse Weather and Lighting Conditions

OBJECTIVES:

Main Objective

To develop an AI-powered weather-adaptive ADAS that improves driving safety by identifying environmental conditions and dynamically adapting driver-assistance behaviour.

Specific Objectives

1.	Detect environmental conditions such as rain, fog, snow, night, and bright sunlight using camera images. 
2.	Classify the current driving condition using a machine-learning/deep-learning model. 
3.	Detect vehicles, pedestrians, road lanes, and obstacles under different environmental conditions. 
4.	Adapt driving-assistance parameters according to the detected condition. 
5.	Increase the following distance during rain, fog, or snow. 
6.	Recommend/reduce safe driving speed under poor visibility. 
7.	Generate fog/rain/night warnings for the driver. 
8.	Improve lane and object detection under difficult lighting/weather conditions. 
9.	Provide a simple real-time dashboard showing detected weather, risk level, and recommended ADAS action. Evaluate the system using accuracy, precision, recall, F1-score and detection performance.
Flow Chart of the Project:
 


Simple decision logic

Detected condition	ADAS response
Normal	Normal assistance
Rain	Reduce recommended speed + increase following distance
Fog	Strong visibility warning + reduce speed + increase distance
Snow	Reduce speed + high-risk warning
Night	Increase detection sensitivity + night warning
Bright sunlight	Glare warning + improve image processing



DATASET:

Dataset 1 — ACDC

Adverse Conditions Dataset with Correspondences.
ACDC is particularly suitable because it was specifically created for understanding driving scenes under adverse conditions.
It contains 4,006 adverse-condition images, approximately divided into:
•	1,000 fog 
•	1,006 nighttime 
•	1,000 rain 
•	1,000 snow 
It also provides corresponding normal-condition images and annotations.
It directly covers the most important conditions in your project:
Rain + Fog + Snow + Night + Normal
Therefore, it is an excellent primary dataset for your weather-adaptive ADAS.


Dataset 2 — BDD100K

BDD100K (Berkeley DeepDrive 100K) is a large-scale driving dataset containing 100,000 driving videos and multiple autonomous-driving tasks. It includes variation in weather, time of day, and driving environments.
It can supplement ACDC with:
•	Daytime scenes 
•	Night scenes 
•	Rainy conditions 
•	Sunny conditions 
•	Overcast conditions 
•	Vehicle detection 
•	Pedestrian detection 
•	Road-scene information 
ACDC will be used as the primary dataset for adverse-weather and low-visibility condition classification, while BDD100K will be used as a supplementary dataset for diverse road scenes, vehicle detection, pedestrian detection, and different lighting/weather conditions.


Methodology overview:

The proposed Smart Weather-Adaptive ADAS uses computer vision and deep learning to identify weather and lighting conditions from vehicle-mounted camera images. The system preprocesses driving images and classifies them into normal, rain, fog, snow, night and bright-sunlight conditions. A road-scene perception module then detects vehicles, pedestrians, obstacles and lane markings. Based on the detected environmental condition and perceived road risk, an adaptive decision engine dynamically changes ADAS recommendations such as safe speed, following distance, warning level and perception sensitivity. The final results are presented through a real-time driver dashboard, enabling the system to provide condition-aware driving assistance.


										Roll Numbers:
								2420030069-A.Raghavasai
								2420090035-B.Ruthwik
								2420030274-M.Deekshith



## 📊 Dataset


This project uses publicly available autonomous driving datasets for weather classification and road-object detection.

### Primary Dataset – ACDC


**ACDC (Adverse Conditions Dataset with Correspondences)** contains driving images captured under adverse weather and visibility conditions such as rain, fog, snow, and nighttime.


- **Dataset:** ACDC – Adverse Conditions Dataset with Correspondences
- **Used for:** Weather/visibility condition detection and semantic segmentation
- **Conditions:** Rain, Fog, Snow, Night
- **Official Dataset:** https://acdc.vision.ee.ethz.ch/

### Secondary Dataset – BDD100K



**BDD100K** is a large-scale driving dataset containing diverse road scenes and annotations for autonomous driving tasks.


- **Dataset:** BDD100K
- **Used for:** Vehicle and pedestrian detection, road scene analysis, and ADAS perception
- **Official Download:** https://bdd-data.berkeley.edu/download.html


### Dataset Usage

The datasets are used only for academic and research purposes. Please refer to the respective dataset websites for their licensing and usage terms.


research paper - 1:

https://www.mdpi.com/2673-2688/3/2/19

research paper - 2:

https://ieeexplore.ieee.org/abstract/document/10596222

