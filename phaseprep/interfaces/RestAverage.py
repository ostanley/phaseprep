from nipype.interfaces.base import BaseInterface, \
    BaseInterfaceInputSpec, traits, File, TraitedSpec
from nipype.utils.filemanip import split_filename
import nibabel as nb
import numpy as np
import os

class RestAverageInputSpec(BaseInterfaceInputSpec):
    func = File(exists=True, desc='functional to analyze', mandatory=True)
    task = traits.Int(desc='length of task, volumes')
    rest = traits.Int(desc='length of rest, volumes')
    trim = traits.Int(desc='number of buffer volumes to remove from rest (allowing return to baseline)', default=1)


class RestAverageOutputSpec(TraitedSpec):
    tsnr = File(exists=True, desc='temporal snr', mandatory=True)
    mean = File(exists=True, desc='mean', mandatory=True)
    noise = File(exists=True, desc='noise', mandatory=True)

class RestAverage(BaseInterface):
    input_spec = RestAverageInputSpec
    output_spec = RestAverageOutputSpec

    def _run_interface(self, runtime):
        # read in data
        fname = self.inputs.func
        _, base, _ = split_filename(fname)

        img = nb.load(fname)
        func = np.array(img.get_data()).astype(float)

        nvols = func.shape[3]

        if self.inputs.rest>0 and self.inputs.task>0:
            #assume block task
            volsrest = np.zeros([1,self.inputs.rest])
            volstask = np.ones([1,self.inputs.task])
            volsrest[:,:trim]=1
            volsrest[:,-trim:]=1

            activity = np.tile(np.concatenate((volsrest, volstask),axis=1)[0], int(np.ceil(float(nvols)/len([volsrest, volstask]))))
            activity = activity[0:nvols]
            activity[0] = 0
            activity[-1] = 0
        else:
            #assume resting state
            activity=np.zeros([nvols,])

        imgmean = np.mean(func[:,:,:,activity!=1], axis=3)
        imgstd = np.std(func[:, :, :, activity!= 1], axis = 3)

        # save figure
        new_img = nb.Nifti1Image(imgmean, img.affine, img.header)
        nb.save(new_img, base + '_mean.nii.gz')
        new_img = nb.Nifti1Image(imgmean/imgstd, img.affine, img.header)
        nb.save(new_img, base + '_tsnr.nii.gz')
        new_img = nb.Nifti1Image(imgstd, img.affine, img.header)
        nb.save(new_img, base + '_noise.nii.gz')
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        fname = self.inputs.func
        _, base, _ = split_filename(fname)
        outputs["tsnr"] = os.path.abspath(base + '_tsnr.nii.gz')
        outputs["mean"] = os.path.abspath(base + '_mean.nii.gz')
        outputs["noise"] = os.path.abspath(base + '_noise.nii.gz')
        return outputs
