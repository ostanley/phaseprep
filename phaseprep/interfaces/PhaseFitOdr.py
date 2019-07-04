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
    sig_ub = traits.Float(desc='Window filter upper bound')
    sig_lb = traits.Float(desc='Window filter lower bound')
    noise_lb = traits.Float(desc='Noise filer lower bound; should be higher than pyiological signal')


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

    def _run_interface(self, runtime):
        f = nb.load(self.inputs.mag)
        mag = f.get_data()

        f = nb.load(self.inputs.phase)
        ph = f.get_data()

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

        linear = odr.unilinear

        # freqs for FT indices
        freqs = np.linspace(-1.0, 1.0, nt) / (2 * self.inputs.TR)

        sig_idx = np.where((freqs > self.inputs.sig_lb) * (freqs < self.inputs.sig_ub))[0]
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
                # if (mag.shape[0] % x)==10000:
                #     print('Processing xVal {} / {}...\n'.format(x, mag.shape[0]))
                # save scalings for investigation
                s1 = stdm[x]
                s2 = stdp[x]

                # now reference notch filtered data
                # make contiguous
                x1 = mag[x, :].copy()
                x2 = ph[x, :].copy()

                mm = np.mean(x1)
                mp = np.mean(x2)

                sm = np.std(x1)
                sp = np.std(x2)

                ests = [sm / sp, mm / mp]

                # and fit model
                # mag = A*phase + B
                # (y=mx+b)
                # call : (x,y,sx,sy)
                mydata = odr.RealData(x2, x1, sx=s2, sy=s1)
                odr_obj = odr.ODR(mydata, linear, beta0=ests, maxit=200)
                res = odr_obj.run()
                est = res.y
                rsq = 1.0 - sum((x1 - est) ** 2) / sum((x1 - mm) ** 2)

                # take out scaled phase signal and re-mean
                sim[x, :] = est

                filt[x, :] = x1 - est + mm
                # estimate R^2
                r2[x] = rsq
                # estimate residuals
                residuals[x, :] = np.sign(x1-est) * (res.delta**2 + res.eps**2)
                delta[x, :] = res.delta
                eps[x, :] = res.eps
                xshift[x, :] = res.xplus

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
