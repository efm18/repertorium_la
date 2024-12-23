# Layout Analysis for Repertorium Project

> **This code is part of REPERTORIUM project, funded by the European Union’s Horizon Europe programme under grant agreement No 101095065.**

This repository is designed to perform layout analysis of the manuscripts from the Repertorium project. This is a demo that includes a simple UI in order to explain graphically the steps to follow.

## Usage

### Step 1: Start the Demo

First, in the root directory, you must run the `python3 -m webui.app`.

### Step 2: Data Manager Screen

- Then, in the `Data manager` screen, the `.json` with the *IIIF manuscripts* must be uploaded (*in this demo, the .json is in the webui/inputs/ directory*), and the *Repertorium format* must be selected. In the `Advanced options` box, a *Package name* must be given, as well as resizing the images to `512` (*this is the size the model is working with*), and the `Train`, `Validation` & `Test` splits must be set to these sizes, correspondingly: `0`, `0`, `1`.
- Once filled all required fields, you can press the `Import` button and go to the next screen.

### Step 3: Predict Layout Analysis Screen

- Next, in the `Predict layout analysis` screen, you must select the package with the name you have written in the previous screen. In order to do so, *in this demo you have to click on the arrow of the directory to open it and then select the `.` folder*. Then, the corresponding model must be selected and the *checkpoint* given (*in this demo, the checkpoint is in the webui/inputs/ directory, and it’s named best.pt*).
- Once filled all required fields, you can press the `Predict` button and go to the next screen.

### Step 4: Show Bounding Boxes

- Finally, in the `Show bounding boxes` screen, the `Show predictions` button must be clicked.
- Then, the package must be selected, the same way than in the previous screen, and the `Generated boxes` should be available.


> **Note:** The models are still in development.
