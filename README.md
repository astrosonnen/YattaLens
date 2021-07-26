# YattaLens

Strong lens finder, described in [Sonnenfeld et al. 2018](https://ui.adsabs.harvard.edu/abs/2018PASJ...70S..29S/abstract).
It consists of two main steps:
1- It looks for tangentially elongated objects, potential arcs, around a supplied set of galaxies.
2- It fits lens models and alternative (non-lens) models to the objects with detecetd arcs.
The second step is considerably slower (on the order of minutes per object), but is only run on those candidates with detected arcs.
The purity of the final sample of candidates varies depending on the properties of the parent sample, but is typically 5-10%.
The main contaminants are spiral galaxies. Pre-selecting the sample with little spiral galaxies helps a lot in reducing the false-positive rate and the total run time.

## Requirements

- SExtractor
- emcee Python package

## Usage

0- Define a YATTADIR environment variable, pointing to the directory where this repository is located.

1- Obtain science image cutouts, variance maps and PSF images for each galaxy in the sample, in each photometric band, and place them in a common directory. YattaLens needs at least two different bands to run, because it uses colors to differentiate between foreground and background objects. Adding a third band can be useful for the production of RGB images for visual inspection. The name of the data files must follow these rules:
- Science image file name: GALAXYID_BAND_sci.fits (where GALAXYID is a unique identifier and BAND is the name of the photometric band)
- Variance map: GALAXYID_BAND_var.fits
- PSF: GALAXYID_BAND_psf.fits

2- Create a .txt file with the list of objects. GALAXYID must be the first column. Any other column will be included in the output file.

3- Create the directories where the output files will be stored. Output files are: models, rgb images and log files.

4- Prepare a configuration file. An example configuration file is the file 'default.yatta' in this repository. 
Important parameters:
CATALOG_FILE: the name of the input file prepared in step 1.
FITBAND: band used for the detection and modelling of the lensed arcs. In SuGOHI I we used the g-band.
LIGHTBAND: band used to model the foreground lens. In SuGOHI I we used the i-band.
COLOR_MAXDIFF: upper limit on the color FITBAND-LIGHTBAND. The default value is 2.0, tailored on g-i. Although lensed sources are usually much bluer than that, the color of the arcs are subject to large errors due to imperfect removal of the foreground light, so 2.0 is a safer limit. It was found to be sufficiently blue to eliminate foregrounds with similar colors as the lens galaxy.

5- Run `$YATTADIR/yatta.py configfile'.

### Output

YattaLens appends the outcome of the analysis on each galaxy to the file defined by the keyword SUMMARY_FILE in the configuration file. Three columns are appended to those already present in the input file. The first column explains the results of the analysis: if YattaLens thinks it is a lens, it writes YATTA. Otherwise, it gives a reason why it thinks it is not a lens. The second column is the image set number corresponding to the most convincing lens model (0 if not a lens. Usually 1, as it is rare that more than one possible image sets are identified). The third column is a data flag. 0: no issues. 1: data not found. 2: image has a different shape from the expected (only if EXPECTED_SIZE keyword is defined in the configuration file).

In MODELDIR, it saves .fits files containing information about the models of the lens candidates (or of all objects, if SAVEALLMODELS is True).

In FIGDIR, it saves .png files of the lens candidates (or of all objects, if MAKEALLFIGS is True). Each figure consists of 10 columns (in the case of a lens): original image, lens-subtracted image, arc and image segmentation map (green=arcs, white=possible images, red=foregrounds), lens model image, source-only model image, image-lens model residual, ring model, image-ring model residual, sersic model, image-sersic model residual.

## Tips

Keep cutout sizes as small as possible: run time scales with the number of pixels.

