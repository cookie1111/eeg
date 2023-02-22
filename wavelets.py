#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.ndimage import gaussian_filter
from scipy.signal import cwt, morlet2

x = np.linspace(0, 1, num=512)
data = np.sin(250 * np.pi * x**2)
data2 = np.cos(130 * np.pi * x**2)

wavelet = 'db2'
level = 4
order = "freq"  # other option is "normal"
interpolation = 'nearest'
cmap = plt.cm.cool

# Construct wavelet packet
wp = pywt.WaveletPacket(data, wavelet, 'symmetric', maxlevel=level)
nodes = wp.get_level(level, order=order)
labels = [n.path for n in nodes]
values = np.array([n.data for n in nodes], 'd')
values = abs(values)

# Show signal and wavelet packet coefficients
fig = plt.figure()
fig.subplots_adjust(hspace=0.2, bottom=.03, left=.07, right=.97, top=.92)
ax = fig.add_subplot(2, 1, 1)
ax.set_title("linchirp signal")
ax.plot(x, data, 'b')
ax.set_xlim(0, x[-1])

ax = fig.add_subplot(2, 1, 2)
ax.set_title("Wavelet packet coefficients at level %d" % level)
ax.imshow(values, interpolation=interpolation, cmap=cmap, aspect="auto",
          origin="lower", extent=[0, 1, 0, len(values)])
ax.set_yticks(np.arange(0.5, len(labels) + 0.5), labels)

# Show spectrogram and wavelet packet coefficients
fig2 = plt.figure()
ax2 = fig2.add_subplot(211)
ax2.specgram(data, NFFT=64, noverlap=32, Fs=2, cmap=cmap,
             interpolation='bilinear')
ax2.set_title("Spectrogram of signal")
ax3 = fig2.add_subplot(212)
ax3.imshow(values, origin='upper', extent=[-1, 1, -1, 1],
           interpolation='nearest')
ax3.set_title("Wavelet packet coefficients")


#plt.show()

#taken from pywavelets ^^^

#scipy implementation with cwt:

a = np.array(cwt(data, morlet2, np.arange(1, 31)))
b = np.array(cwt(data2, morlet2, np.arange(1, 31)))
print(pywt.cwt(data, np.arange(1, 41), 'cmor1.5-1.0')[0].shape)
print(a[0,0])
print(a.shape)

# Cross-wavelet sepctrum 
cs_spect = np.multiply(a,np.conjugate(b))
print(cs_spect[0,0])

# smoothing is performed by a weighted moving average in both dimensions (time and frequency)

# use a gaussian filter along both axis from the scipy library!

#print(cs_spect[0,0]) TEST

# all of the operations are element wise!!!
# coherence is: abs(Smooth(cs_spec_xy))^2/(smooth(abs(cs_spec_x)^2)*smooth(abs(cs_spec_y)^2))

upper = np.square(np.abs(gaussian_filter(cs_spect,(4,4))))
#lower = gaussian_filter(np.square(np.abs(

print(cs_spect.shape)
