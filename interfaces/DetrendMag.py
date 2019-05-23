from nipype.interfaces.base import BaseInterface, \
    BaseInterfaceInputSpec, traits, File, TraitedSpec
from nipype.utils.filemanip import split_filename
import nibabel as nb
import numpy as np
import os

class DetrendMagInputSpec(BaseInterfaceInputSpec):
    mag = File(exists=True, desc='magnitude image to detrend', mandatory=True)

class DetrendMagOutputSpec(TraitedSpec):
    detrended_mag = File(exists=True, desc="detrended_mag")

class DetrendMag(BaseInterface):
    input_spec = DetrendMagInputSpec
    output_spec = DetrendMagOutputSpec

    def _run_interface(self, runtime):
        fname = self.inputs.mag
        img = nb.load(fname)
        data = np.array(img.get_data())

        detrendmag = np.zeros_like(data)
        for x in range(data.shape[0]):
            for y in range(data.shape[1]):
                for z in range(data.shape[2]):
                    # detrend mag without demeaning it
                    xval = np.linspace(0, len(data[x, y, z, :]) - 1, len(data[x, y, z, :]))
                    detrendmag[x, y, z, :] = data[x, y, z, :] - \
                        np.polyval(np.polyfit(xval, data[x, y, z, :], 1), xval) + \
                        np.mean(data[x, y, z, :])

        new_img = nb.Nifti1Image(detrendmag, img.affine, img.header)
        _, base, _ = split_filename(fname)
        nb.save(new_img, base + '_DetrendMag.nii.gz')

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        fname = self.inputs.mag
        _, base, _ = split_filename(fname)
        outputs["detrended_mag"] = os.path.abspath(base + '_DetrendMag.nii.gz')
        return outputs
