from nipype.interfaces.base import BaseInterface, \
    BaseInterfaceInputSpec, traits, File, TraitedSpec
from nipype.utils.filemanip import split_filename
import nibabel as nb
import numpy as np
import os

class PreprocessPhaseInputSpec(BaseInterfaceInputSpec):
    phase = File(exists=True, desc='phase image', mandatory=True)


class PreprocessPhaseOutputSpec(TraitedSpec):
    delta_phase = File(exists=True, desc="delta_phase")
    uw_phase = File(exists=True, desc="uw_phase")
    detrended_phase = File(exists=True, desc="detrended_phase")

class PreprocessPhase(BaseInterface):
    input_spec = PreprocessPhaseInputSpec
    output_spec = PreprocessPhaseOutputSpec

    def _run_interface(self, runtime):
        fname = self.inputs.phase
        img = nb.load(fname)
        data = np.array(img.get_data()).astype(float)

        phaseuw = np.zeros_like(data, dtype=np.float)
        deltaphase = np.zeros_like(data, dtype=np.float)
        detrendphase = np.zeros_like(data, dtype=np.float)
        for x in range(data.shape[0]):
            for y in range(data.shape[1]):
                for z in range(data.shape[2]):
                    # create delta phase time series
                    deltaphase[x, y, z, :] = data[x, y, z, :] - data[x, y, z, 0] * np.ones_like(data[x, y, z, :])
                    # unwrap phase
                    phaseuw[x, y, z, :] = np.unwrap(deltaphase[x, y, z, :])
                    # detrend phase
                    xval = np.linspace(0, len(phaseuw[x, y, z, :]) - 1, len(phaseuw[x, y, z, :]))
                    detrendphase[x, y, z, :] = phaseuw[x, y, z, :] - \
                        np.polyval(np.polyfit(xval, phaseuw[x, y, z, :], 1), xval) + \
                        np.mean(phaseuw[x, y, z, :])

        new_img = nb.Nifti1Image(deltaphase, img.affine, img.header)
        _, base, _ = split_filename(fname)
        nb.save(new_img, base + '_deltaphase.nii.gz')
        new_img = nb.Nifti1Image(phaseuw, img.affine, img.header)
        _, base, _ = split_filename(fname)
        nb.save(new_img, base + '_uwphase.nii.gz')
        new_img = nb.Nifti1Image(detrendphase, img.affine, img.header)
        _, base, _ = split_filename(fname)
        nb.save(new_img, base + '_detrendphase.nii.gz')

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        fname = self.inputs.phase
        _, base, _ = split_filename(fname)
        outputs["detrended_phase"] = os.path.abspath(base + '_detrendphase.nii.gz')
        outputs["uw_phase"] = os.path.abspath(base + '_uwphase.nii.gz')
        outputs["delta_phase"] = os.path.abspath(base + '_deltaphase.nii.gz')
        return outputs
