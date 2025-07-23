# st-depth-estimation
Temporary repo for the development of an depth estimation tool for AddaxAI

It will be incorporated into the existing AddaxAI streamlit repo eventually, but its easier for now to just see it as a separate tool. We'll tie it together later.

## Installation

1. Clone the repository.

```bash
git clone https://github.com/PetervanLunteren/st-depth-estimation.git
```

2. Install the dependencies:

```bash
pip install -r st-depth-estimation/requirements.txt
```

## Usage

To run the app, use the following command:

```bash
streamlit run st-depth-estimation/app.py
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
1. let the user select a representative image to do the calibration. Perhaps using https://github.com/jrieke/streamlit-image-select ?
2. run the depth anything model (https://github.com/DepthAnything/Depth-Anything-V2) over that image and get the distance map for that image. It would be nice to have the RGB image next to the coloured distance map of that image. Perhaps something like this https://huggingface.co/spaces/depth-anything/Depth-Anything-V2 ?

![Alt text](assets/example.png)

3. do the actual calibration, where the user clicks on a few objects, and inputs the known distance. E.g., this tree is 3 meters away, this bush is 10 meters away, etc. How many? I don't know. We'll have to test. It would be nice to be able to click on either the RGB image or the depth map if the user perfers either one, and have calibration point show up at both images. Perhaps using https://image-coordinates.streamlit.app/dynamic_update ?
4. calculate the distances of all the bounding boxes of the animals in the other images in that deployment, so we can add it to the results. We can do different things here, like the distance of the center of the bounding box (perhaps not useful for weirdly shaped animals that dont always have a body in the center like snakes, giraffes, etc?). Maybe best to take the center of the bottom of the bbox, so that we have the distance at the ground level. This is also error prone due to grass or other object in the way.. We'll have to give it some trial and error... We can also caluclate the hight of the animal, which will give the user some information on the age group of the animal (juvenile / adult).

# Closing remarks

This is just an initial idea! Please don't take these to strict. In my experience it always turns out different that you would expect, so please feel free to play around and try out what feels best!

