import os
import scipy
import numpy as np
import matplotlib.pyplot as plt
from pydicom import dcmread

from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

INPUT_FOLDER = "./dicoms/Teste Dicon4"
patients = os.listdir(INPUT_FOLDER)
patients.sort()


def load_scan(path):
    print("Loading scan", path)
    slices = [dcmread(path + "/" + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    if slices[0].ImagePositionPatient[2] == slices[1].ImagePositionPatient[2]:
        sec_num = 2
        while (
            slices[0].ImagePositionPatient[2] == slices[sec_num].ImagePositionPatient[2]
        ):
            sec_num = sec_num + 1
        slice_num = int(len(slices) / sec_num)
        slices.sort(key=lambda x: float(x.InstanceNumber))
        slices = slices[0:slice_num]
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    try:
        slice_thickness = np.abs(
            slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2]
        )
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices  # list of DICOM


DICOM_PATH = "./dicoms/Teste Dicon4/20190607/S0000000286"

patient_scan = load_scan(DICOM_PATH)
print(patient_scan[0])


def get_pixels_hu(slices):
    image = np.stack(
        [s.pixel_array for s in slices if s.pixel_array.shape == (260, 260)]
    )
    image = image.astype(np.int16)

    image[image == -2000] = 0

    for slice_i in range(len(image)):
        intercept = slices[slice_i].RescaleIntercept
        slope = slices[slice_i].RescaleSlope

        if slope != 1:
            image[slice_i] = slope * image[slice_i].astype(np.float64)
            image[slice_i] = image[slice_i].astype(np.int16)

        image[slice_i] += np.int16(intercept)

    return np.array(image, dtype=np.int16)


pixels = get_pixels_hu(patient_scan)
print(pixels)

plt.hist(pixels.flatten(), bins=80, color="c")
plt.xlabel("Hounsfield Units (HU)")
plt.ylabel("Frequency")
plt.show()

# Show some slice in the middle
plt.imshow(pixels[50], cmap=plt.cm.gray)
plt.show()


def resample(image, scan, new_spacing=[1, 1, 1]):
    # Determine current pixel spacing
    spacing = np.array(
        [scan[0].SliceThickness] + list(scan[0].PixelSpacing), dtype=np.float32
    )

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode="nearest")

    return image, new_spacing


pix_resampled, spacing = resample(pixels, patient_scan, [1, 1, 1])
print("Shape before resampling\t", pixels.shape)
print("Shape after resampling\t", pix_resampled.shape)


def plot_3d(image, threshold=-300):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2, 1, 0)

    verts, faces, normals, values = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()


plot_3d(pix_resampled, 400)
