
# FFT_DTensor
A function calculating multi-dimensional Fourier Transform for DTensors. DTensor is an extension to TensorFlow for synchronous distributed computing (https://www.tensorflow.org/guide/dtensor_overview)

## Input and Output

Input is a DTensor with any dimension. An example is shown below:

	input = tf.complex(tf.random.stateless_normal(shape=(1, 2, 4), seed=(1, 2), dtype=tf.float32),

	tf.random.stateless_normal(shape=(1, 2, 4), seed=(2, 4), dtype=tf.float32))

	layout = dtensor.Layout(['x', 'y', 'z'], mesh)

	input = relayout_complex(input, layout=layout)

Output is a DTensor with the same dimension as the input. Users can relayout the output by defining the parameter "final_layout".

## Parameters
* axes：Int, optional
    The number of axes to compute the FFT. If not given, the FFT will happen on all axes.

* norm： {“backward”, “ortho”, “forward”}, optional
    Normalization mode (see numpy.fft). Default is “backward”. Indicates which direction of the forward/backward pair of transforms is scaled and with what normalization factor.

* output_format: {"transposed", "regular"}, optional
    Default is "transposed". Indicates if the output will be transposed after calculation.
    
 *  axis1, axis2: {'1', 'None'}, optional
	 If not given and needed for transpose, the default is "-1", "-2". Demonstrates which axes are used if "output_format" is set as "transposed".
	 

## Example
	output = fftnd_dtensor(input, mesh, axes=1, output_format='regular')
	print(output)
 
## Acknowlegement
This work is guided by Yu Feng(github: rainwoodman@). 
