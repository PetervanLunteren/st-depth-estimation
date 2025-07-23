# st-depth-estimation
Temporary repo for the development of an depth estimation tool for AddaxAI

It will be incorporated into the existing AddaxAI streamlit repo eventually, but its easier for now to just see it as a separate tool. We'll tie it together later.

## Installation

1. Clone the repository.
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

To run the app, use the following command:

```bash
streamlit run app.py
```

## Idea
The idea is that users can add an estimation of the distance from the camera to the animal. This then can provide information on the size of the animal. When users have reached this tool, they have already processed the images with AI recognition models, so the bounding box of the animals, and the prediction are known. 

My idea was to run the depth anything model (https://github.com/DepthAnything/Depth-Anything-V2) over one image of the deployment, calibrate it a few points, and then be able to calculate the distances of each animal.

You can find an example deployment in this repo: /test-imgs/
In that folder there is a CSV file with all the prediciton and bbox information: /test-imgs/results_detections.csv

<details>
<summary>example detections CSV</summary>

<br>

| relative_path | label                         | confidence | bbox_left | bbox_top | bbox_right | bbox_bottom | DateTimeOriginal    | Latitude           | Longitude           |
|---------------|-------------------------------|------------|-----------|----------|------------|-------------|---------------------|--------------------|---------------------|
| img_0001.jpg  | species Alcelaphus buselaphus | 0.92241    | 558       | 398      | 941        | 654         | 18/01/2013 08:58    | 0.27805552777777776| 36.87395458333334   |
| img_0001.jpg  | family Bovidae                | 0.94621    | 1058      | 468      | 1227       | 574         | 18/01/2013 08:58    | 0.27805552777777776| 36.87395458333334   |
| img_0002.jpg  | species Alcelaphus buselaphus | 0.97218    | 552       | 397      | 919        | 652         | 18/01/2013 08:58    | 0.27805552777777776| 36.87395458333334   |

</details>
																															


# UI

I dont really have a concrete idea of how it should look, but I believe the following should be present
* first step is to let the user select a representative image to do the calibration
* second step is to do the actual calibration, where the user clicks on a few objects, and inputs the known distance. E.g., this tree is 3 meters away, this bush is 10 meters away, etc. 
* third step would be to run the depth anything model (https://github.com/DepthAnything/Depth-Anything-V2) over that image and get the distance map for that image.
* fourth step would be to calculate the distances of all the bounding boxes of the animals in the other images in that deployment, so we can add it to the results. 