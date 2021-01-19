# 用于将dicom序列导出为nii文件并将大小降采样为128*128*128
# nii生成drr代码为C++ 需要在VS中调试
import numpy as np
from scipy.ndimage import zoom
import SimpleITK as sitk
import pydicom
import os
import dicom2nifti.patch_pydicom_encodings
from tqdm import tqdm

dicom2nifti.patch_pydicom_encodings.apply()
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def ImageResample(sitk_image, new_spacing=[1.0, 1.0, 1.0], is_label=False):
    '''
    sitk_image:
    new_spacing: x,y,z
    is_label: if True, using Interpolator `sitk.sitkNearestNeighbor`
    '''
    size = np.array(sitk_image.GetSize())
    spacing = np.array(sitk_image.GetSpacing())
    new_spacing = np.array(new_spacing)

    new_size = size * spacing / new_spacing
    new_spacing_refine = size * spacing / new_size
    new_spacing_refine = [float(s) for s in new_spacing_refine]
    new_size = [int(s) for s in new_size]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputDirection(sitk_image.GetDirection())
    resample.SetOutputOrigin(sitk_image.GetOrigin())
    resample.SetSize(new_size)
    resample.SetOutputSpacing(new_spacing_refine)

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        # resample.SetInterpolator(sitk.sitkBSpline)
        resample.SetInterpolator(sitk.sitkLinear)

    newimage = resample.Execute(sitk_image)
    return newimage



def plot_3d(image, threshold=-300):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    image = image.astype(np.int16)
    p = image.transpose(2, 1, 0)
    #     p = p[:,:,::-1]

    print(p.shape)
    verts, faces, _, x = measure.marching_cubes_lewiner(p, threshold)  # marching_cubes_classic measure.marching_cubes

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()


raw_ct_dir = "D:\\7_15data\\raw_ct\\0"
output = "D:\\7_15data\\new_ct\\0.nii"

reader = sitk.ImageSeriesReader()
img_names = reader.GetGDCMSeriesFileNames(raw_ct_dir)

reader.SetFileNames(img_names)
ds = reader.Execute()

img_array = sitk.GetArrayFromImage(ds)
np.shape(img_array)

print(ds.GetOrigin())
print(ds.GetSize())
print(ds.GetSpacing())
print(ds.GetDirection())
print(ds.GetDimension())
print(ds.GetWidth())
print(ds.GetHeight())
print(ds.GetDepth())
print(ds.GetPixelIDValue())
print(ds.GetPixelIDTypeAsString())
print(ds.GetNumberOfComponentsPerPixel())

nor = ImageResample(ds)
print(nor.GetSize())
print(nor.GetSpacing())
sitk.WriteImage(nor, 'new_resample.nii')
# nor_arr = sitk.GetArrayFromImage(nor)

# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (10, 5))
# ax1.imshow(img_array[150,:,:], cmap=plt.cm.bone)
# ax1.set_title('T')
# ax2.imshow(img_array[:,150,:], cmap=plt.cm.bone)
# ax2.set_title('C')
# ax3.imshow(img_array[:,:,100], cmap=plt.cm.bone)
# ax3.set_title('S')

# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (10, 5))
# ax1.imshow(nor_arr[150,:,:], cmap=plt.cm.bone)
# ax1.set_title('T')
# ax2.imshow(nor_arr[:,150,:], cmap=plt.cm.bone)
# ax2.set_title('C')
# ax3.imshow(nor_arr[:,:,100], cmap=plt.cm.bone)
# ax3.set_title('S')
# nor_arr[nor_arr > 0] = 255
# nor_arr[nor_arr <= 0] = 0
# plot_3d(nor_arr, 100)
# plt.show()
