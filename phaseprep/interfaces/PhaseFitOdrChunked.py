from nipype.interfaces.base import BaseInterface, \
    BaseInterfaceInputSpec, traits, File, TraitedSpec
from nipype.utils.filemanip import split_filename
from threadpoolctl import threadpool_limits
from scipy import odr as odr
import nibabel as nb
import numpy as np
import os
import uuid
import os.path as path


class PhaseFitOdrChunkedInputSpec(BaseInterfaceInputSpec):
    phase = File(exists=True, desc='phase image', mandatory=True)
    mag = File(exists=True, desc='mag image', mandatory=True)
    TR = traits.Float(desc='repetition time of the scan', mandatory=True)
    noise_lb = traits.Float(desc='Noise filter lower bound; should be higher than pyiological signal')
    global_regressors = File(exists=True, desc='File of regressors in .tsv format')

class PhaseFitOdrChunkedOutputSpec(TraitedSpec):
    sim = File(exists=True, desc="sim")
    filt = File(exists=True, desc="filt")
    corr = File(exists=True, desc="corr")
    stdm = File(exists=True, desc="std error in mag")
    stdp = File(exists=True, desc="std error in phase")
    fit_coeff = File(exists=True, desc="std error in phase")

class PhaseFitOdrChunked(BaseInterface):
    input_spec = PhaseFitOdrChunkedInputSpec
    output_spec = PhaseFitOdrChunkedOutputSpec

    def multiplelinear(self, beta, x):
        f=np.zeros(x[0].shape)
        for r in range(x.shape[0]):
            f+=beta[r]*x[r]
        f+=beta[-1]*np.ones(x[0].shape)
        return f

    def _run_interface(self, runtime):
        with threadpool_limits(limits=1, user_api='blas'):
            #load in images and get shape
            fmag = nb.load(self.inputs.mag)
            fphase = nb.load(self.inputs.phase)

            saveshape = fmag.header.get_data_shape()
            nt = saveshape[-1]
            savetype=fmag.header.get_data_dtype()
            memory_limit=2*(10**9) #2GB of data max used in bytes

            # calculate maximum number of voxel in memory at one time
            num_voxels_in_chunk = np.round(0.95*memory_limit/(2*nt*savetype.itemsize),-3)

            #this interface is gonna go slice by slice as the dataobj cannot be reshaped
            assert(num_voxels_in_chunk>np.prod(saveshape[0:2]))

            # create output memmaps
            saveuuid=uuid.uuid4()
            filesim = path.join(os.getcwd(), 'filesim_'+str(saveuuid)+'.dat')
            sim = np.memmap(filesim, dtype=savetype, shape=tuple(saveshape), mode='w+')
            filefilt = path.join(os.getcwd(), 'filefilt_'+str(saveuuid)+'.dat')
            filt = np.memmap(filefilt, dtype=savetype, shape=tuple(saveshape), mode='w+')
            filestdm = path.join(os.getcwd(), 'filestdm_'+str(saveuuid)+'.dat')
            stdm = np.memmap(filestdm, dtype=savetype, shape=tuple(saveshape[0:-1]), mode='w+')
            filestdp = path.join(os.getcwd(), 'filestdp_'+str(saveuuid)+'.dat')
            stdp = np.memmap(filestdp, dtype=savetype, shape=tuple(saveshape[0:-1]), mode='w+')
            filer2 = path.join(os.getcwd(), 'filer2_'+str(saveuuid)+'.dat')
            r2 = np.memmap(filer2, dtype=savetype, shape=tuple(saveshape[0:-1]), mode='w+')
            filebeta = path.join(os.getcwd(), 'filebeta_'+str(saveuuid)+'.dat')
            # save beta intialization until we know size

            #load additional regressors (motion or physio)
            if self.inputs.global_regressors:
                regressors = np.loadtxt(self.inputs.global_regressors)
                beta = np.memmap(filebeta, dtype=savetype, shape=tuple([saveshape[0], saveshape[1],
                                                                       saveshape[2], 2+regressors.shape[1]]), mode='w+')
            else:
                beta = np.memmap(filebeta, dtype=savetype, shape=tuple([saveshape[0], saveshape[1],
                                                                       saveshape[2], 2]), mode='w+')

            #initialize model
            linearfit = odr.Model(self.multiplelinear)

            # freqs for FT indices used for noise estimation
            freqs = np.linspace(-1.0, 1.0, nt) / (2 * self.inputs.TR)
            noise_idx = np.where((abs(freqs) > self.inputs.noise_lb))[0]
            noise_mask = np.fft.fftshift(1.0 * (abs(freqs) > self.inputs.noise_lb))

            # create mask that is 3% of initial image intensity
            mag_first = np.reshape(np.array(fmag.dataobj[:,:,:,0]), [-1,])
            mask = np.reshape(mag_first > 0.03 * np.max(mag_first), [-1,saveshape[2]])

            # load image in one slice at a time
            for slice in range(saveshape[2]):
                print('On slice: ', slice, ' / ', saveshape[2])
                mag=np.reshape(fmag.dataobj[:,:,slice,:], [-1,nt])
                ph=np.reshape(fphase.dataobj[:,:,slice,:], [-1,nt])

                # Estimate sigmas in one preproc step as fft can be done across multiple voxels at once
                temp = mag
                mu = np.mean(mag, axis=-1)
                stdm[:,:,slice] = np.reshape(np.std(np.fft.ifft(np.fft.fft(mag - mu[..., np.newaxis]) * noise_mask), -1),
                                              [saveshape[0], saveshape[1]])
                temp = ph
                mu = np.mean(ph, axis=-1)
                stdp[:,:,slice] = np.reshape(np.std(np.fft.ifft(np.fft.fft(ph - mu[..., np.newaxis]) * noise_mask), -1),
                                             [saveshape[0], saveshape[1]])

                #perform fit voxel by voxel
                for x in range(mag.shape[0]):
                    if mask[x,slice]:
                        #need row and column index for saving as memmaps are 4D
                        r,c = np.unravel_index(x, saveshape[0:2])

                        mm = np.mean(mag[x,:])
                        mp = np.mean(ph[x,:])

                        # intialize X as columns of data, regressors (optional), intercept
                        if 'regressors' in locals():
                            design=np.row_stack((ph[x,:], regressors.T, np.ones(ph[x,:].shape)))
                            ests = np.hstack([[stdm[r,c,slice] / stdp[r,c,slice]],
                                             np.ones((regressors.shape[1],)), [mm / mp]])
                            mydata = odr.RealData(design, mag[x,:].T,sx=np.hstack((stdp[r,c,slice],
                                                  np.std(regressors, axis=0),1)), sy=stdm[r,c,slice])
                        else:
                            design=np.row_stack((ph[x,:], np.ones(ph[x,:].shape)))
                            ests = [stdm[r,c,slice] / stdp[r,c,slice], mm / mp]
                            mydata = odr.RealData(design, mag[x,:].T, sx=np.hstack([stdp[r,c,slice],1]),
                                                  sy=stdm[r,c,slice])
                        # and fit model
                        # mag = A*phase + B*regressors + C
                        # (y=mx+b)
                        # call : (x,y,sx,sy)
                        odr_obj = odr.ODR(mydata, linearfit, beta0=ests, maxit=200)
                        res = odr_obj.run()
                        est = res.y
                        r2[r,c,slice] = 1.0 - sum((mag[x,:] - est) ** 2) / sum((mag[x,:] - mm) ** 2)
                        beta[r,c,slice,:] = res.beta

                        # take out scaled phase signal
                        sim[r,c,slice] = ph[x,:]*res.beta[0]
                        filt[r,c,slice] = mag[x,:] - est + mm

            _, outname, _ = split_filename(self.inputs.mag)
            print(outname)
            outnii = nb.Nifti1Image(sim, affine=fmag.affine, header=fmag.get_header())
            outnii.to_filename(outname + '_sim.nii.gz')
            outnii = nb.Nifti1Image(filt, affine=fmag.affine, header=fmag.get_header())
            outnii.to_filename(outname + '_filt.nii.gz')

            # plot fit statistic info
            outnii = nb.Nifti1Image(stdp, affine=fmag.affine, header=fmag.get_header())
            outnii.to_filename(outname + '_stdp.nii.gz')
            outnii = nb.Nifti1Image(stdm, affine=fmag.affine, header=fmag.get_header())
            outnii.to_filename(outname + '_stdm.nii.gz')
            outnii = nb.Nifti1Image(r2, affine=fmag.affine, header=fmag.get_header())
            outnii.to_filename(outname + '_r2.nii.gz')
            outnii = nb.Nifti1Image(beta, affine=fmag.affine, header=fmag.get_header())
            outnii.to_filename(outname + '_betas.nii.gz')
            return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        _, base, _ = split_filename(self.inputs.mag)
        outputs["sim"] = os.path.abspath(base + '_sim.nii.gz')
        outputs["filt"] = os.path.abspath(base + '_filt.nii.gz')
        outputs["corr"] = os.path.abspath(base + '_r2.nii.gz')
        outputs["fit_coeff"] = os.path.abspath(base + '_betas.nii.gz')
        outputs["stdm"] = os.path.abspath(base + '_stdm.nii.gz')
        outputs["stdp"] = os.path.abspath(base + '_stdp.nii.gz')
        return outputs
