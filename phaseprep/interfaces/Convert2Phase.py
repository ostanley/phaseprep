from nipype.interfaces.base import BaseInterface, \
    BaseInterfaceInputSpec, traits, File, TraitedSpec
from nipype.utils.filemanip import split_filename
import nibabel as nb
import numpy as np
import os
import math

#Converts from real and imaginary to phase using atan2 to avoid sign abiguity
#Mimics behaviour of fslcomplex not fslmaths

class Convert2PhaseInputSpec(BaseInterfaceInputSpec):
    real_image = File(exists=True, desc='real image', mandatory=True)
    imaginary_image = File(exists=True, desc='phase image', mandatory=True)

class Convert2PhaseOutputSpec(TraitedSpec):
    phase_image = File(exists=True, desc="phase image")

class Convert2Phase(BaseInterface):
    input_spec = Convert2PhaseInputSpec
    output_spec = Convert2PhaseOutputSpec

    def _run_interface(self, runtime):
        real_img = nb.load(self.inputs.real_image)
        real_data = np.array(real_img.get_data()).astype(float)
        imaginary_data = np.array(nb.load(self.inputs.imaginary_image).get_data()).astype(float)

        phase_data = np.arctan2(imaginary_data, real_data)

        new_img = nb.Nifti1Image(phase_data, real_img.affine, real_img.header)
        _, base, _ = split_filename(self.inputs.real_image)
        nb.save(new_img, base + '_phase.nii.gz')

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        fname = self.inputs.real_image
        _, base, _ = split_filename(fname)
        outputs["phase_image"] = os.path.abspath(base + '_phase.nii.gz')
        return outputs
