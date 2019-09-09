from nipype.interfaces.base import BaseInterface, \
    BaseInterfaceInputSpec, traits, File, TraitedSpec
from nipype.utils.filemanip import split_filename
from scipy import odr as odr
import nibabel as nb
import numpy as np
import os

class PhaseFitOdrInputSpec(BaseInterfaceInputSpec):
    phase = File(exists=True, desc='phase image', mandatory=True)
    mag = File(exists=True, desc='mag image', mandatory=True)
    TR = traits.Float(desc='repetition time of the scan', mandatory=True)
    noise_lb = traits.Float(desc='Noise filter lower bound; should be higher than pyiological signal')
    global_regressors = File(exists=True, desc='File of regressors in .tsv format')

class PhaseFitOdrOutputSpec(TraitedSpec):
    sim = File(exists=True, desc="sim")
    filt = File(exists=True, desc="filt")
    corr = File(exists=True, desc="corr")
    residuals = File(exists=True, desc="residuals")
    resx = File(exists=True, desc="residuals x only")
    resy = File(exists=True, desc="residuals y only")
    stdm = File(exists=True, desc="std error in mag")
    stdp = File(exists=True, desc="std error in phase")

class PhaseFitOdr(BaseInterface):
    input_spec = PhaseFitOdrInputSpec
    output_spec = PhaseFitOdrOutputSpec

    def multiplelinear(self, beta, x):
        f=np.zeros(x[0].shape)
        for r in range(x.shape[0]):
            f+=beta[r]*x[r]
        f+=beta[-1]*np.ones(x[0].shape)
        return f

    def _run_interface(self, runtime):
        f = nb.load(self.inputs.mag)
        mag = f.get_data()

        f = nb.load(self.inputs.phase)
        ph = f.get_data()

        if self.inputs.global_regressors:
            regressors = np.loadtxt(self.inputs.global_regressors)

        saveshape = np.array(mag.shape)
        nt = mag.shape[-1]

        scales = np.zeros(np.prod(saveshape[0:-1]))
        filt = np.zeros((np.prod(saveshape[0:-1]),nt))
        sim = np.zeros_like(filt)
        residuals = np.zeros_like(filt)

        delta = np.zeros_like(filt)
        eps = np.zeros_like(filt)
        xshift = np.zeros_like(filt)
        stdm = np.zeros(saveshape[0:-1])
        stdp = np.zeros(saveshape[0:-1])
        r2 = np.zeros_like(scales)

        mag = np.array(mag)
        mm = np.mean(mag, axis=-1)
        mask = mm > 0.03 * np.max(mm)

        linearfit = odr.Model(self.multiplelinear)

        # freqs for FT indices
        freqs = np.linspace(-1.0, 1.0, nt) / (2 * self.inputs.TR)

        noise_idx = np.where((abs(freqs) > self.inputs.noise_lb))[0]
        noise_mask = np.fft.fftshift(1.0 * (abs(freqs) > self.inputs.noise_lb))

        # Estimate sigmas in one preproc step
        for x in range(mag.shape[0]):
            temp = mag[x, :, :, :]
            mu = np.mean(temp, -1)
            stdm[x, :, :] = np.std(np.fft.ifft(np.fft.fft(temp - mu[..., np.newaxis]) * noise_mask), -1)
            temp = ph[x, :, :, :]
            mu = np.mean(temp, -1)
            stdp[x, :, :] = np.std(np.fft.ifft(np.fft.fft(temp - mu[..., np.newaxis]) * noise_mask), -1)

        stdm=np.reshape(stdm, (-1,))
        stdp=np.reshape(stdp, (-1,))
        mask=np.reshape(mask, (-1,))
        mag=np.reshape(mag, (-1, nt))
        ph=np.reshape(ph, (-1, nt))
        for x in range(mag.shape[0]):
            if mask[x]:
                mm = np.mean(mag[x,:])
                mp = np.mean(ph[x,:])

                if 'regressors' in locals():
                    design=np.row_stack((ph[x,:], regressors.T, np.ones(ph[x,:].shape)))
                    ests = np.hstack([[stdm[x] / stdp[x]], np.ones((regressors.shape[1],)), [mm / mp]])
                    mydata = odr.RealData(design, mag[x,:].T,sx=np.hstack((stdp[x], np.std(regressors, axis=0),1)), sy=stdm[x])
                else:
                    design=np.row_stack((ph[x,:], np.ones(ph[x,:].shape)))
                    ests = [stdm[x] / stdp[x], mm / mp]
                    mydata = odr.RealData(design, mag[x,:].T, sx=np.hstack([stdp[x],1]), sy=stdm[x])
                print(mydata.x.shape, mydata.y.shape, mydata.sx.shape, mydata.sy, len(ests), design.shape)

                # and fit model
                # mag = A*phase + B*regressors + C
                # (y=mx+b)
                # call : (x,y,sx,sy)
                odr_obj = odr.ODR(mydata, linearfit, beta0=ests, maxit=200)
                res = odr_obj.run()
                print(res.stopreason, res.beta)
                est = res.y
                r2[x] = 1.0 - sum((mag[x,:] - est) ** 2) / sum((mag[x,:] - mm) ** 2)

                # take out scaled phase signal and re-mean may need correction
                sim[x, :] = ph[x,:]*res.beta[0]

                filt[x, :] = mag[x,:] - est + mm
                # estimate residuals
                residuals[x, :] = np.sign(mag[x,:]-est) * (np.sum(res.delta**2, axis=0) + res.eps**2)
                delta[x, :] = np.sum(res.delta, axis=0)
                eps[x, :] = res.eps
                xshift[x, :] = np.sum(res.xplus, axis=0)

        _, outname, _ = split_filename(self.inputs.mag)
        print(outname)
        outnii = nb.Nifti1Image(np.reshape(sim, saveshape), affine=f.affine, header=f.get_header())
        outnii.to_filename(outname + '_sim.nii.gz')
        outnii = nb.Nifti1Image(np.reshape(filt, saveshape), affine=f.affine, header=f.get_header())
        outnii.to_filename(outname + '_filt.nii.gz')
        outnii = nb.Nifti1Image(np.reshape(residuals, saveshape), affine=f.affine, header=f.get_header())
        outnii.to_filename(outname + '_residuals.nii.gz')
        outnii = nb.Nifti1Image(np.reshape(delta, saveshape), affine=f.affine, header=f.get_header())
        outnii.to_filename(outname + '_xres.nii.gz')
        outnii = nb.Nifti1Image(np.reshape(eps, saveshape), affine=f.affine, header=f.get_header())
        outnii.to_filename(outname + '_yres.nii.gz')
        outnii = nb.Nifti1Image(np.reshape(xshift, saveshape), affine=f.affine, header=f.get_header())
        outnii.to_filename(outname + '_xplus.nii.gz')

        # plot fit statistic info
        outnii = nb.Nifti1Image(np.reshape(stdp, saveshape[0:-1]), affine=f.affine, header=f.get_header())
        outnii.to_filename(outname + '_stdp.nii.gz')
        outnii = nb.Nifti1Image(np.reshape(stdm, saveshape[0:-1]), affine=f.affine, header=f.get_header())
        outnii.to_filename(outname + '_stdm.nii.gz')
        outnii = nb.Nifti1Image(np.reshape(r2, saveshape[0:-1]), affine=f.affine, header=f.get_header())
        outnii.to_filename(outname + '_r2.nii.gz')
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        _, base, _ = split_filename(self.inputs.mag)
        outputs["sim"] = os.path.abspath(base + '_sim.nii.gz')
        outputs["filt"] = os.path.abspath(base + '_filt.nii.gz')
        outputs["corr"] = os.path.abspath(base + '_r2.nii.gz')
        outputs["residuals"] = os.path.abspath(base + '_residuals.nii.gz')
        outputs["resx"] = os.path.abspath(base + '_xres.nii.gz')
        outputs["resy"] = os.path.abspath(base + '_yres.nii.gz')
        outputs["stdm"] = os.path.abspath(base + '_stdm.nii.gz')
        outputs["stdp"] = os.path.abspath(base + '_stdp.nii.gz')
        return outputs
