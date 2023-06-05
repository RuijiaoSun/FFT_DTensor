import tensorflow as tf
import numpy as np
from tensorflow.experimental import dtensor

# This function workes for CPU, GPU, or TPU
device_type = dtensor.preferred_device_type()
if device_type == 'CPU':
  cpu = tf.config.list_physical_devices(device_type)
  tf.config.set_logical_device_configuration(cpu[0], [tf.config.LogicalDeviceConfiguration()] * 8)
if device_type == 'GPU':
  gpu = tf.config.list_physical_devices(device_type)
  tf.config.set_logical_device_configuration(gpu[0], [tf.config.LogicalDeviceConfiguration(memory_limit=1000)] * 8)
dtensor.initialize_accelerator_system()

# relayout for complex numbers
def relayout_complex(input, layout):
  t2real = tf.math.real(input)
  t2imag = tf.math.imag(input)
  t2real = dtensor.relayout(t2real, layout=layout)
  t2imag = dtensor.relayout(t2imag, layout=layout)
  t2 = tf.complex(t2real, t2imag)
  return t2

# Find the number of unsharded axes
def number_unsharded(init_layout):
  l = init_layout.sharding_specs
  unsharded_axis = []
  for i in range(len(l)):
    if l[i] == dtensor.UNSHARDED:
      unsharded_axis.append(i)
  return unsharded_axis

# Define the new layout for swapped axes
def redefine_layout(init_layout, now_layout, mesh, ax1, ax2, reshard=False, swap_second=False):
    if ax1 > ax2:
      ax1, ax2 = ax2, ax1
    # Reshard
    if reshard == True and init_layout is not None:
      l = init_layout.sharding_specs
      unsharded_axis = number_unsharded(init_layout)
      if len(unsharded_axis) == 0:
        l[ax1] = dtensor.UNSHARDED
      else:
        if swap_second:
          l[ax1], l[unsharded_axis[1]] = l[unsharded_axis[1]], l[ax1]
        else:
          l[ax1], l[unsharded_axis[0]] = l[unsharded_axis[0]], l[ax1]

      # Swap
      l[ax1], l[ax2] = l[ax2], l[ax1]
    else:
      l = now_layout.sharding_specs
      l[ax1], l[ax2] = l[ax2], l[ax1]
    return dtensor.Layout(l, mesh)

# Swap axes and redefine the layout
def transpose_layout(input, now_layout, mesh, ax1, ax2, reshard=False, init_layout=None):
  # Swap
  dtensor_arrray = tf.experimental.numpy.swapaxes(input, ax1, ax2)
  # relayout
  dtensor_arrray = relayout_complex(dtensor_arrray, layout = redefine_layout(init_layout, now_layout, mesh, ax1, ax2, reshard=False))
  return dtensor_arrray

# The newest version introduces fft2d() to better speed up. The swapaxes and redefining the layout happen twice
def transpose_layout2d(input, now_layout, mesh, ax1, ax2, ax3, ax4, reshard=False, init_layout=None):
  # Swap
  dtensor_arrray = tf.experimental.numpy.swapaxes(input, ax1, ax4)
  dtensor_arrray = relayout_complex(dtensor_arrray, layout = redefine_layout(init_layout, now_layout, mesh, ax1, ax4, reshard=False))
  init_layout = dtensor.fetch_layout(dtensor_arrray)
  dtensor_arrray = tf.experimental.numpy.swapaxes(input, ax2, ax3)
  dtensor_arrray = relayout_complex(dtensor_arrray, layout = redefine_layout(init_layout, now_layout, mesh, ax2, ax3, reshard=False, swap_second=True))
  # relayout
  return dtensor_arrray

# Main fftnd() function. The necessary parameters are the input dtensor array and the mesh.
def fftnd(input, mesh, axes=None, norm=None, output_format='transposed', axis1=None, axis2=None):
  if axes == None:
      axes = len(input.shape)

  init_layout = dtensor.fetch_layout(input)
  l = init_layout.sharding_specs
  print("init_layout: "+str(init_layout))

  # Count how many UNSHARDED axes the input has.
  unsharded_axis = number_unsharded(init_layout)
  print(unsharded_axis)
  counts = len(unsharded_axis)  
  print(counts)

  # Set one of the axes to be UNSHARDED for fully sharded dtensors.
  if counts < 2:
    if counts == 0:
      l[-1] = dtensor.UNSHARDED
      unsharded_axis.append(-1)
    elif counts == 1:
      l[unsharded_axis[0]], l[-1] = l[-1], l[unsharded_axis[0]]

    # Copy input using now_layout
    now_layout = dtensor.Layout(l, mesh)
    output = relayout_complex(input, layout=now_layout)

    ax = -1
    while axes > 0:
      output = transpose_layout(output, now_layout, mesh, ax, -1, reshard=True, init_layout=init_layout)
      output = tf.signal.fft(output)
      output = transpose_layout(output, now_layout, mesh, ax, -1)
      ax -= 1
      axes -= 1

  else:
      l[unsharded_axis[0]], l[-1] = l[-1], l[unsharded_axis[0]]
      l[unsharded_axis[1]], l[-2] = l[-2], l[unsharded_axis[1]]
      # Copy input using now_layout
      now_layout = dtensor.Layout(l, mesh)
      output = relayout_complex(input, layout=now_layout)
      ax = -1
      while axes > 1:
        output = transpose_layout2d(output, now_layout, mesh, ax, ax - 1, unsharded_axis[0], unsharded_axis[1], reshard=True, init_layout=init_layout)
        output = tf.signal.fft2d(output)
        print("2D fft introduced!")
        output = transpose_layout2d(output, now_layout, mesh, ax, ax - 1, unsharded_axis[0], unsharded_axis[1])
        ax -= 2
        axes -= 2
      # Do fft1d for the last axis
      output = transpose_layout(output, now_layout, mesh, ax, -1, reshard=True, init_layout=init_layout)
      output = tf.signal.fft(output)
      output = transpose_layout(output, now_layout, mesh, ax, -1)

  # Relayout to the initial status
  output = relayout_complex(output, layout=init_layout)


  # Normalization https://numpy.org/doc/stable/reference/routines.fft.html#normalization
  # Find how many points we have.
  N = 1
  for i in range(len(output.shape)):
    N *= output.shape[i]
  if norm == 'ortho':
    output *= 1 / np.sqrt(N)
  elif norm == 'forward':
    output *= 1 / N

  if output_format == 'transposed':
    if axis1 == None:
      axis1 = axes[-2]
    if axis2 == None:
      axis2 = axes[-1]
    output = transpose_layout(output, now_layout, mesh, axis1, axis2)

  return input

# Run a test
mesh = dtensor.create_distributed_mesh(mesh_dims=[('x', 2), ('y', 2), ('z', 2)], device_type=device_type)
input = tf.complex(
    tf.random.stateless_normal(shape=(2, 2, 4), seed=(1, 2), dtype=tf.float32),
    tf.random.stateless_normal(shape=(2, 2, 4), seed=(2, 4), dtype=tf.float32))
input_np = input.numpy()

init_layout = dtensor.Layout(['x', 'y', dtensor.UNSHARDED], mesh)
readable_layout = dtensor.Layout([dtensor.UNSHARDED, dtensor.UNSHARDED, dtensor.UNSHARDED], mesh)

# Run a test
input = relayout_complex(input, layout=init_layout)
print("Layout for the input: " + str(dtensor.fetch_layout(input))+"\n\n")
output_fftnd = fftnd(input, mesh, output_format='regular')
output_fftnd = relayout_complex(output_fftnd, layout=readable_layout)
print("\n---Result from personalized FFTnd--")
print("Real part: \n"+str(tf.math.real(output_fftnd)))
print("Imag part: \n"+str(tf.math.imag(output_fftnd)))

# Compare the result from our fftnd() and numpy.fft.fft()
output_np = np.fft.fft(input_np)
print("\n---Result from Numpy.FFT---")
print(output_np)


