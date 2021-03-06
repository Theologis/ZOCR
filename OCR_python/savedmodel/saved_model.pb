??
?:?9
,
Abs
x"T
y"T"
Ttype:

2	
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	??
?
	ApplyAdam
var"T?	
m"T?	
v"T?
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T?" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
?
ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
x
Assign
ref"T?

value"T

output_ref"T?"	
Ttype"
validate_shapebool("
use_lockingbool(?
s
	AssignSub
ref"T?

value"T

output_ref"T?" 
Ttype:
2	"
use_lockingbool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
Conv2DBackpropFilter

input"T
filter_sizes
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
B
Equal
x"T
y"T
z
"
Ttype:
2	
?
,
Exp
x"T
y"T"
Ttype:

2
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	
?
FusedBatchNorm
x"T

scale"T
offset"T	
mean"T
variance"T
y"T

batch_mean"T
batch_variance"T
reserve_space_1"T
reserve_space_2"T"
Ttype:
2"
epsilonfloat%??8"-
data_formatstringNHWC:
NHWCNCHW"
is_trainingbool(
?
FusedBatchNormGrad

y_backprop"T
x"T

scale"T
reserve_space_1"T
reserve_space_2"T

x_backprop"T
scale_backprop"T
offset_backprop"T
reserve_space_3"T
reserve_space_4"T"
Ttype:
2"
epsilonfloat%??8"-
data_formatstringNHWC:
NHWCNCHW"
is_trainingbool(
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
,
Log
x"T
y"T"
Ttype:

2
.
Log1p
x"T
y"T"
Ttype:

2
?

LogSoftmax
logits"T

logsoftmax"T"
Ttype:
2
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
?
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
;
Maximum
x"T
y"T
z"T"
Ttype:

2	?
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
?
OneHot
indices"TI	
depth
on_value"T
	off_value"T
output"T"
axisint?????????"	
Ttype"
TItype0	:
2	
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
)
Rank

input"T

output"	
Ttype
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
5

Reciprocal
x"T
y"T"
Ttype:

2	
E
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
/
Sign
x"T
y"T"
Ttype:

2	
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
j
SoftmaxCrossEntropyWithLogits
features"T
labels"T	
loss"T
backprop"T"
Ttype:
2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
2
StopGradient

input"T
output"T"	
Ttype
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
-
Tanh
x"T
y"T"
Ttype:

2
:
TanhGrad
y"T
dy"T
z"T"
Ttype:

2
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype?"
shapeshape"
dtypetype"
	containerstring "
shared_namestring ?
&
	ZerosLike
x"T
y"T"	
Ttype"serve*1.12.02
b'unknown'??
[
Variable/initial_valueConst*
valueB
 *o?:*
dtype0*
_output_shapes
: 
l
Variable
VariableV2*
dtype0*
shared_name *
shape: *
	container *
_output_shapes
: 
?
Variable/AssignAssignVariableVariable/initial_value*
T0*
use_locking(*
validate_shape(*
_class
loc:@Variable*
_output_shapes
: 
a
Variable/readIdentityVariable*
T0*
_class
loc:@Variable*
_output_shapes
: 
}

input_realPlaceholder*
dtype0*$
shape:?????????  */
_output_shapes
:?????????  
j
input_zPlaceholder*
dtype0*
shape:?????????d*'
_output_shapes
:?????????d
F
yPlaceholder*
dtype0*
shape:*
_output_shapes
:
O

label_maskPlaceholder*
dtype0*
shape:*
_output_shapes
:
T
drop_rate/inputConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
f
	drop_ratePlaceholderWithDefaultdrop_rate/input*
dtype0*
shape: *
_output_shapes
: 
?
7generator/dense/kernel/Initializer/random_uniform/shapeConst*
valueB"d      *
dtype0*)
_class
loc:@generator/dense/kernel*
_output_shapes
:
?
5generator/dense/kernel/Initializer/random_uniform/minConst*
valueB
 *?zX?*
dtype0*)
_class
loc:@generator/dense/kernel*
_output_shapes
: 
?
5generator/dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *?zX=*
dtype0*)
_class
loc:@generator/dense/kernel*
_output_shapes
: 
?
?generator/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform7generator/dense/kernel/Initializer/random_uniform/shape*
T0*
dtype0*
seed2 *

seed *)
_class
loc:@generator/dense/kernel*
_output_shapes
:	d?
?
5generator/dense/kernel/Initializer/random_uniform/subSub5generator/dense/kernel/Initializer/random_uniform/max5generator/dense/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@generator/dense/kernel*
_output_shapes
: 
?
5generator/dense/kernel/Initializer/random_uniform/mulMul?generator/dense/kernel/Initializer/random_uniform/RandomUniform5generator/dense/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@generator/dense/kernel*
_output_shapes
:	d?
?
1generator/dense/kernel/Initializer/random_uniformAdd5generator/dense/kernel/Initializer/random_uniform/mul5generator/dense/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@generator/dense/kernel*
_output_shapes
:	d?
?
generator/dense/kernel
VariableV2*
dtype0*
shared_name *
shape:	d?*
	container *)
_class
loc:@generator/dense/kernel*
_output_shapes
:	d?
?
generator/dense/kernel/AssignAssigngenerator/dense/kernel1generator/dense/kernel/Initializer/random_uniform*
T0*
use_locking(*
validate_shape(*)
_class
loc:@generator/dense/kernel*
_output_shapes
:	d?
?
generator/dense/kernel/readIdentitygenerator/dense/kernel*
T0*)
_class
loc:@generator/dense/kernel*
_output_shapes
:	d?
?
6generator/dense/bias/Initializer/zeros/shape_as_tensorConst*
valueB:?*
dtype0*'
_class
loc:@generator/dense/bias*
_output_shapes
:
?
,generator/dense/bias/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*'
_class
loc:@generator/dense/bias*
_output_shapes
: 
?
&generator/dense/bias/Initializer/zerosFill6generator/dense/bias/Initializer/zeros/shape_as_tensor,generator/dense/bias/Initializer/zeros/Const*
T0*

index_type0*'
_class
loc:@generator/dense/bias*
_output_shapes	
:?
?
generator/dense/bias
VariableV2*
dtype0*
shared_name *
shape:?*
	container *'
_class
loc:@generator/dense/bias*
_output_shapes	
:?
?
generator/dense/bias/AssignAssigngenerator/dense/bias&generator/dense/bias/Initializer/zeros*
T0*
use_locking(*
validate_shape(*'
_class
loc:@generator/dense/bias*
_output_shapes	
:?
?
generator/dense/bias/readIdentitygenerator/dense/bias*
T0*'
_class
loc:@generator/dense/bias*
_output_shapes	
:?
?
generator/dense/MatMulMatMulinput_zgenerator/dense/kernel/read*
T0*
transpose_b( *
transpose_a( *(
_output_shapes
:??????????
?
generator/dense/BiasAddBiasAddgenerator/dense/MatMulgenerator/dense/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:??????????
p
generator/Reshape/shapeConst*%
valueB"????      ?   *
dtype0*
_output_shapes
:
?
generator/ReshapeReshapegenerator/dense/BiasAddgenerator/Reshape/shape*
T0*
Tshape0*0
_output_shapes
:??????????
?
4generator/batch_normalization/gamma/Initializer/onesConst*
valueB?*  ??*
dtype0*6
_class,
*(loc:@generator/batch_normalization/gamma*
_output_shapes	
:?
?
#generator/batch_normalization/gamma
VariableV2*
dtype0*
shared_name *
shape:?*
	container *6
_class,
*(loc:@generator/batch_normalization/gamma*
_output_shapes	
:?
?
*generator/batch_normalization/gamma/AssignAssign#generator/batch_normalization/gamma4generator/batch_normalization/gamma/Initializer/ones*
T0*
use_locking(*
validate_shape(*6
_class,
*(loc:@generator/batch_normalization/gamma*
_output_shapes	
:?
?
(generator/batch_normalization/gamma/readIdentity#generator/batch_normalization/gamma*
T0*6
_class,
*(loc:@generator/batch_normalization/gamma*
_output_shapes	
:?
?
4generator/batch_normalization/beta/Initializer/zerosConst*
valueB?*    *
dtype0*5
_class+
)'loc:@generator/batch_normalization/beta*
_output_shapes	
:?
?
"generator/batch_normalization/beta
VariableV2*
dtype0*
shared_name *
shape:?*
	container *5
_class+
)'loc:@generator/batch_normalization/beta*
_output_shapes	
:?
?
)generator/batch_normalization/beta/AssignAssign"generator/batch_normalization/beta4generator/batch_normalization/beta/Initializer/zeros*
T0*
use_locking(*
validate_shape(*5
_class+
)'loc:@generator/batch_normalization/beta*
_output_shapes	
:?
?
'generator/batch_normalization/beta/readIdentity"generator/batch_normalization/beta*
T0*5
_class+
)'loc:@generator/batch_normalization/beta*
_output_shapes	
:?
?
;generator/batch_normalization/moving_mean/Initializer/zerosConst*
valueB?*    *
dtype0*<
_class2
0.loc:@generator/batch_normalization/moving_mean*
_output_shapes	
:?
?
)generator/batch_normalization/moving_mean
VariableV2*
dtype0*
shared_name *
shape:?*
	container *<
_class2
0.loc:@generator/batch_normalization/moving_mean*
_output_shapes	
:?
?
0generator/batch_normalization/moving_mean/AssignAssign)generator/batch_normalization/moving_mean;generator/batch_normalization/moving_mean/Initializer/zeros*
T0*
use_locking(*
validate_shape(*<
_class2
0.loc:@generator/batch_normalization/moving_mean*
_output_shapes	
:?
?
.generator/batch_normalization/moving_mean/readIdentity)generator/batch_normalization/moving_mean*
T0*<
_class2
0.loc:@generator/batch_normalization/moving_mean*
_output_shapes	
:?
?
>generator/batch_normalization/moving_variance/Initializer/onesConst*
valueB?*  ??*
dtype0*@
_class6
42loc:@generator/batch_normalization/moving_variance*
_output_shapes	
:?
?
-generator/batch_normalization/moving_variance
VariableV2*
dtype0*
shared_name *
shape:?*
	container *@
_class6
42loc:@generator/batch_normalization/moving_variance*
_output_shapes	
:?
?
4generator/batch_normalization/moving_variance/AssignAssign-generator/batch_normalization/moving_variance>generator/batch_normalization/moving_variance/Initializer/ones*
T0*
use_locking(*
validate_shape(*@
_class6
42loc:@generator/batch_normalization/moving_variance*
_output_shapes	
:?
?
2generator/batch_normalization/moving_variance/readIdentity-generator/batch_normalization/moving_variance*
T0*@
_class6
42loc:@generator/batch_normalization/moving_variance*
_output_shapes	
:?
f
#generator/batch_normalization/ConstConst*
valueB *
dtype0*
_output_shapes
: 
h
%generator/batch_normalization/Const_1Const*
valueB *
dtype0*
_output_shapes
: 
?
,generator/batch_normalization/FusedBatchNormFusedBatchNormgenerator/Reshape(generator/batch_normalization/gamma/read'generator/batch_normalization/beta/read#generator/batch_normalization/Const%generator/batch_normalization/Const_1*
is_training(*
T0*
data_formatNHWC*
epsilon%o?:*L
_output_shapes:
8:??????????:?:?:?:?
j
%generator/batch_normalization/Const_2Const*
valueB
 *?p}?*
dtype0*
_output_shapes
: 
?
3generator/batch_normalization/AssignMovingAvg/sub/xConst*
valueB
 *  ??*
dtype0*<
_class2
0.loc:@generator/batch_normalization/moving_mean*
_output_shapes
: 
?
1generator/batch_normalization/AssignMovingAvg/subSub3generator/batch_normalization/AssignMovingAvg/sub/x%generator/batch_normalization/Const_2*
T0*<
_class2
0.loc:@generator/batch_normalization/moving_mean*
_output_shapes
: 
?
3generator/batch_normalization/AssignMovingAvg/sub_1Sub.generator/batch_normalization/moving_mean/read.generator/batch_normalization/FusedBatchNorm:1*
T0*<
_class2
0.loc:@generator/batch_normalization/moving_mean*
_output_shapes	
:?
?
1generator/batch_normalization/AssignMovingAvg/mulMul3generator/batch_normalization/AssignMovingAvg/sub_11generator/batch_normalization/AssignMovingAvg/sub*
T0*<
_class2
0.loc:@generator/batch_normalization/moving_mean*
_output_shapes	
:?
?
-generator/batch_normalization/AssignMovingAvg	AssignSub)generator/batch_normalization/moving_mean1generator/batch_normalization/AssignMovingAvg/mul*
T0*
use_locking( *<
_class2
0.loc:@generator/batch_normalization/moving_mean*
_output_shapes	
:?
?
5generator/batch_normalization/AssignMovingAvg_1/sub/xConst*
valueB
 *  ??*
dtype0*@
_class6
42loc:@generator/batch_normalization/moving_variance*
_output_shapes
: 
?
3generator/batch_normalization/AssignMovingAvg_1/subSub5generator/batch_normalization/AssignMovingAvg_1/sub/x%generator/batch_normalization/Const_2*
T0*@
_class6
42loc:@generator/batch_normalization/moving_variance*
_output_shapes
: 
?
5generator/batch_normalization/AssignMovingAvg_1/sub_1Sub2generator/batch_normalization/moving_variance/read.generator/batch_normalization/FusedBatchNorm:2*
T0*@
_class6
42loc:@generator/batch_normalization/moving_variance*
_output_shapes	
:?
?
3generator/batch_normalization/AssignMovingAvg_1/mulMul5generator/batch_normalization/AssignMovingAvg_1/sub_13generator/batch_normalization/AssignMovingAvg_1/sub*
T0*@
_class6
42loc:@generator/batch_normalization/moving_variance*
_output_shapes	
:?
?
/generator/batch_normalization/AssignMovingAvg_1	AssignSub-generator/batch_normalization/moving_variance3generator/batch_normalization/AssignMovingAvg_1/mul*
T0*
use_locking( *@
_class6
42loc:@generator/batch_normalization/moving_variance*
_output_shapes	
:?
T
generator/mul/xConst*
valueB
 *??L>*
dtype0*
_output_shapes
: 
?
generator/mulMulgenerator/mul/x,generator/batch_normalization/FusedBatchNorm*
T0*0
_output_shapes
:??????????
?
generator/MaximumMaximumgenerator/mul,generator/batch_normalization/FusedBatchNorm*
T0*0
_output_shapes
:??????????
?
Bgenerator/conv2d_transpose/kernel/Initializer/random_uniform/shapeConst*%
valueB"      @   ?   *
dtype0*4
_class*
(&loc:@generator/conv2d_transpose/kernel*
_output_shapes
:
?
@generator/conv2d_transpose/kernel/Initializer/random_uniform/minConst*
valueB
 *???*
dtype0*4
_class*
(&loc:@generator/conv2d_transpose/kernel*
_output_shapes
: 
?
@generator/conv2d_transpose/kernel/Initializer/random_uniform/maxConst*
valueB
 *??=*
dtype0*4
_class*
(&loc:@generator/conv2d_transpose/kernel*
_output_shapes
: 
?
Jgenerator/conv2d_transpose/kernel/Initializer/random_uniform/RandomUniformRandomUniformBgenerator/conv2d_transpose/kernel/Initializer/random_uniform/shape*
T0*
dtype0*
seed2 *

seed *4
_class*
(&loc:@generator/conv2d_transpose/kernel*'
_output_shapes
:@?
?
@generator/conv2d_transpose/kernel/Initializer/random_uniform/subSub@generator/conv2d_transpose/kernel/Initializer/random_uniform/max@generator/conv2d_transpose/kernel/Initializer/random_uniform/min*
T0*4
_class*
(&loc:@generator/conv2d_transpose/kernel*
_output_shapes
: 
?
@generator/conv2d_transpose/kernel/Initializer/random_uniform/mulMulJgenerator/conv2d_transpose/kernel/Initializer/random_uniform/RandomUniform@generator/conv2d_transpose/kernel/Initializer/random_uniform/sub*
T0*4
_class*
(&loc:@generator/conv2d_transpose/kernel*'
_output_shapes
:@?
?
<generator/conv2d_transpose/kernel/Initializer/random_uniformAdd@generator/conv2d_transpose/kernel/Initializer/random_uniform/mul@generator/conv2d_transpose/kernel/Initializer/random_uniform/min*
T0*4
_class*
(&loc:@generator/conv2d_transpose/kernel*'
_output_shapes
:@?
?
!generator/conv2d_transpose/kernel
VariableV2*
dtype0*
shared_name *
shape:@?*
	container *4
_class*
(&loc:@generator/conv2d_transpose/kernel*'
_output_shapes
:@?
?
(generator/conv2d_transpose/kernel/AssignAssign!generator/conv2d_transpose/kernel<generator/conv2d_transpose/kernel/Initializer/random_uniform*
T0*
use_locking(*
validate_shape(*4
_class*
(&loc:@generator/conv2d_transpose/kernel*'
_output_shapes
:@?
?
&generator/conv2d_transpose/kernel/readIdentity!generator/conv2d_transpose/kernel*
T0*4
_class*
(&loc:@generator/conv2d_transpose/kernel*'
_output_shapes
:@?
?
1generator/conv2d_transpose/bias/Initializer/zerosConst*
valueB@*    *
dtype0*2
_class(
&$loc:@generator/conv2d_transpose/bias*
_output_shapes
:@
?
generator/conv2d_transpose/bias
VariableV2*
dtype0*
shared_name *
shape:@*
	container *2
_class(
&$loc:@generator/conv2d_transpose/bias*
_output_shapes
:@
?
&generator/conv2d_transpose/bias/AssignAssigngenerator/conv2d_transpose/bias1generator/conv2d_transpose/bias/Initializer/zeros*
T0*
use_locking(*
validate_shape(*2
_class(
&$loc:@generator/conv2d_transpose/bias*
_output_shapes
:@
?
$generator/conv2d_transpose/bias/readIdentitygenerator/conv2d_transpose/bias*
T0*2
_class(
&$loc:@generator/conv2d_transpose/bias*
_output_shapes
:@
q
 generator/conv2d_transpose/ShapeShapegenerator/Maximum*
T0*
out_type0*
_output_shapes
:
x
.generator/conv2d_transpose/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
z
0generator/conv2d_transpose/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
z
0generator/conv2d_transpose/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
(generator/conv2d_transpose/strided_sliceStridedSlice generator/conv2d_transpose/Shape.generator/conv2d_transpose/strided_slice/stack0generator/conv2d_transpose/strided_slice/stack_10generator/conv2d_transpose/strided_slice/stack_2*
Index0*
end_mask *
shrink_axis_mask*
T0*

begin_mask *
new_axis_mask *
ellipsis_mask *
_output_shapes
: 
z
0generator/conv2d_transpose/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:
|
2generator/conv2d_transpose/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
|
2generator/conv2d_transpose/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
*generator/conv2d_transpose/strided_slice_1StridedSlice generator/conv2d_transpose/Shape0generator/conv2d_transpose/strided_slice_1/stack2generator/conv2d_transpose/strided_slice_1/stack_12generator/conv2d_transpose/strided_slice_1/stack_2*
Index0*
end_mask *
shrink_axis_mask*
T0*

begin_mask *
new_axis_mask *
ellipsis_mask *
_output_shapes
: 
z
0generator/conv2d_transpose/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
|
2generator/conv2d_transpose/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
|
2generator/conv2d_transpose/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
*generator/conv2d_transpose/strided_slice_2StridedSlice generator/conv2d_transpose/Shape0generator/conv2d_transpose/strided_slice_2/stack2generator/conv2d_transpose/strided_slice_2/stack_12generator/conv2d_transpose/strided_slice_2/stack_2*
Index0*
end_mask *
shrink_axis_mask*
T0*

begin_mask *
new_axis_mask *
ellipsis_mask *
_output_shapes
: 
b
 generator/conv2d_transpose/mul/yConst*
value	B :*
dtype0*
_output_shapes
: 
?
generator/conv2d_transpose/mulMul*generator/conv2d_transpose/strided_slice_1 generator/conv2d_transpose/mul/y*
T0*
_output_shapes
: 
d
"generator/conv2d_transpose/mul_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
?
 generator/conv2d_transpose/mul_1Mul*generator/conv2d_transpose/strided_slice_2"generator/conv2d_transpose/mul_1/y*
T0*
_output_shapes
: 
d
"generator/conv2d_transpose/stack/3Const*
value	B :@*
dtype0*
_output_shapes
: 
?
 generator/conv2d_transpose/stackPack(generator/conv2d_transpose/strided_slicegenerator/conv2d_transpose/mul generator/conv2d_transpose/mul_1"generator/conv2d_transpose/stack/3*

axis *
T0*
N*
_output_shapes
:
?
+generator/conv2d_transpose/conv2d_transposeConv2DBackpropInput generator/conv2d_transpose/stack&generator/conv2d_transpose/kernel/readgenerator/Maximum*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(*/
_output_shapes
:?????????@
?
"generator/conv2d_transpose/BiasAddBiasAdd+generator/conv2d_transpose/conv2d_transpose$generator/conv2d_transpose/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:?????????@
?
6generator/batch_normalization_1/gamma/Initializer/onesConst*
valueB@*  ??*
dtype0*8
_class.
,*loc:@generator/batch_normalization_1/gamma*
_output_shapes
:@
?
%generator/batch_normalization_1/gamma
VariableV2*
dtype0*
shared_name *
shape:@*
	container *8
_class.
,*loc:@generator/batch_normalization_1/gamma*
_output_shapes
:@
?
,generator/batch_normalization_1/gamma/AssignAssign%generator/batch_normalization_1/gamma6generator/batch_normalization_1/gamma/Initializer/ones*
T0*
use_locking(*
validate_shape(*8
_class.
,*loc:@generator/batch_normalization_1/gamma*
_output_shapes
:@
?
*generator/batch_normalization_1/gamma/readIdentity%generator/batch_normalization_1/gamma*
T0*8
_class.
,*loc:@generator/batch_normalization_1/gamma*
_output_shapes
:@
?
6generator/batch_normalization_1/beta/Initializer/zerosConst*
valueB@*    *
dtype0*7
_class-
+)loc:@generator/batch_normalization_1/beta*
_output_shapes
:@
?
$generator/batch_normalization_1/beta
VariableV2*
dtype0*
shared_name *
shape:@*
	container *7
_class-
+)loc:@generator/batch_normalization_1/beta*
_output_shapes
:@
?
+generator/batch_normalization_1/beta/AssignAssign$generator/batch_normalization_1/beta6generator/batch_normalization_1/beta/Initializer/zeros*
T0*
use_locking(*
validate_shape(*7
_class-
+)loc:@generator/batch_normalization_1/beta*
_output_shapes
:@
?
)generator/batch_normalization_1/beta/readIdentity$generator/batch_normalization_1/beta*
T0*7
_class-
+)loc:@generator/batch_normalization_1/beta*
_output_shapes
:@
?
=generator/batch_normalization_1/moving_mean/Initializer/zerosConst*
valueB@*    *
dtype0*>
_class4
20loc:@generator/batch_normalization_1/moving_mean*
_output_shapes
:@
?
+generator/batch_normalization_1/moving_mean
VariableV2*
dtype0*
shared_name *
shape:@*
	container *>
_class4
20loc:@generator/batch_normalization_1/moving_mean*
_output_shapes
:@
?
2generator/batch_normalization_1/moving_mean/AssignAssign+generator/batch_normalization_1/moving_mean=generator/batch_normalization_1/moving_mean/Initializer/zeros*
T0*
use_locking(*
validate_shape(*>
_class4
20loc:@generator/batch_normalization_1/moving_mean*
_output_shapes
:@
?
0generator/batch_normalization_1/moving_mean/readIdentity+generator/batch_normalization_1/moving_mean*
T0*>
_class4
20loc:@generator/batch_normalization_1/moving_mean*
_output_shapes
:@
?
@generator/batch_normalization_1/moving_variance/Initializer/onesConst*
valueB@*  ??*
dtype0*B
_class8
64loc:@generator/batch_normalization_1/moving_variance*
_output_shapes
:@
?
/generator/batch_normalization_1/moving_variance
VariableV2*
dtype0*
shared_name *
shape:@*
	container *B
_class8
64loc:@generator/batch_normalization_1/moving_variance*
_output_shapes
:@
?
6generator/batch_normalization_1/moving_variance/AssignAssign/generator/batch_normalization_1/moving_variance@generator/batch_normalization_1/moving_variance/Initializer/ones*
T0*
use_locking(*
validate_shape(*B
_class8
64loc:@generator/batch_normalization_1/moving_variance*
_output_shapes
:@
?
4generator/batch_normalization_1/moving_variance/readIdentity/generator/batch_normalization_1/moving_variance*
T0*B
_class8
64loc:@generator/batch_normalization_1/moving_variance*
_output_shapes
:@
h
%generator/batch_normalization_1/ConstConst*
valueB *
dtype0*
_output_shapes
: 
j
'generator/batch_normalization_1/Const_1Const*
valueB *
dtype0*
_output_shapes
: 
?
.generator/batch_normalization_1/FusedBatchNormFusedBatchNorm"generator/conv2d_transpose/BiasAdd*generator/batch_normalization_1/gamma/read)generator/batch_normalization_1/beta/read%generator/batch_normalization_1/Const'generator/batch_normalization_1/Const_1*
is_training(*
T0*
data_formatNHWC*
epsilon%o?:*G
_output_shapes5
3:?????????@:@:@:@:@
l
'generator/batch_normalization_1/Const_2Const*
valueB
 *?p}?*
dtype0*
_output_shapes
: 
?
5generator/batch_normalization_1/AssignMovingAvg/sub/xConst*
valueB
 *  ??*
dtype0*>
_class4
20loc:@generator/batch_normalization_1/moving_mean*
_output_shapes
: 
?
3generator/batch_normalization_1/AssignMovingAvg/subSub5generator/batch_normalization_1/AssignMovingAvg/sub/x'generator/batch_normalization_1/Const_2*
T0*>
_class4
20loc:@generator/batch_normalization_1/moving_mean*
_output_shapes
: 
?
5generator/batch_normalization_1/AssignMovingAvg/sub_1Sub0generator/batch_normalization_1/moving_mean/read0generator/batch_normalization_1/FusedBatchNorm:1*
T0*>
_class4
20loc:@generator/batch_normalization_1/moving_mean*
_output_shapes
:@
?
3generator/batch_normalization_1/AssignMovingAvg/mulMul5generator/batch_normalization_1/AssignMovingAvg/sub_13generator/batch_normalization_1/AssignMovingAvg/sub*
T0*>
_class4
20loc:@generator/batch_normalization_1/moving_mean*
_output_shapes
:@
?
/generator/batch_normalization_1/AssignMovingAvg	AssignSub+generator/batch_normalization_1/moving_mean3generator/batch_normalization_1/AssignMovingAvg/mul*
T0*
use_locking( *>
_class4
20loc:@generator/batch_normalization_1/moving_mean*
_output_shapes
:@
?
7generator/batch_normalization_1/AssignMovingAvg_1/sub/xConst*
valueB
 *  ??*
dtype0*B
_class8
64loc:@generator/batch_normalization_1/moving_variance*
_output_shapes
: 
?
5generator/batch_normalization_1/AssignMovingAvg_1/subSub7generator/batch_normalization_1/AssignMovingAvg_1/sub/x'generator/batch_normalization_1/Const_2*
T0*B
_class8
64loc:@generator/batch_normalization_1/moving_variance*
_output_shapes
: 
?
7generator/batch_normalization_1/AssignMovingAvg_1/sub_1Sub4generator/batch_normalization_1/moving_variance/read0generator/batch_normalization_1/FusedBatchNorm:2*
T0*B
_class8
64loc:@generator/batch_normalization_1/moving_variance*
_output_shapes
:@
?
5generator/batch_normalization_1/AssignMovingAvg_1/mulMul7generator/batch_normalization_1/AssignMovingAvg_1/sub_15generator/batch_normalization_1/AssignMovingAvg_1/sub*
T0*B
_class8
64loc:@generator/batch_normalization_1/moving_variance*
_output_shapes
:@
?
1generator/batch_normalization_1/AssignMovingAvg_1	AssignSub/generator/batch_normalization_1/moving_variance5generator/batch_normalization_1/AssignMovingAvg_1/mul*
T0*
use_locking( *B
_class8
64loc:@generator/batch_normalization_1/moving_variance*
_output_shapes
:@
V
generator/mul_1/xConst*
valueB
 *??L>*
dtype0*
_output_shapes
: 
?
generator/mul_1Mulgenerator/mul_1/x.generator/batch_normalization_1/FusedBatchNorm*
T0*/
_output_shapes
:?????????@
?
generator/Maximum_1Maximumgenerator/mul_1.generator/batch_normalization_1/FusedBatchNorm*
T0*/
_output_shapes
:?????????@
?
Dgenerator/conv2d_transpose_1/kernel/Initializer/random_uniform/shapeConst*%
valueB"          @   *
dtype0*6
_class,
*(loc:@generator/conv2d_transpose_1/kernel*
_output_shapes
:
?
Bgenerator/conv2d_transpose_1/kernel/Initializer/random_uniform/minConst*
valueB
 *??L?*
dtype0*6
_class,
*(loc:@generator/conv2d_transpose_1/kernel*
_output_shapes
: 
?
Bgenerator/conv2d_transpose_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *??L=*
dtype0*6
_class,
*(loc:@generator/conv2d_transpose_1/kernel*
_output_shapes
: 
?
Lgenerator/conv2d_transpose_1/kernel/Initializer/random_uniform/RandomUniformRandomUniformDgenerator/conv2d_transpose_1/kernel/Initializer/random_uniform/shape*
T0*
dtype0*
seed2 *

seed *6
_class,
*(loc:@generator/conv2d_transpose_1/kernel*&
_output_shapes
: @
?
Bgenerator/conv2d_transpose_1/kernel/Initializer/random_uniform/subSubBgenerator/conv2d_transpose_1/kernel/Initializer/random_uniform/maxBgenerator/conv2d_transpose_1/kernel/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@generator/conv2d_transpose_1/kernel*
_output_shapes
: 
?
Bgenerator/conv2d_transpose_1/kernel/Initializer/random_uniform/mulMulLgenerator/conv2d_transpose_1/kernel/Initializer/random_uniform/RandomUniformBgenerator/conv2d_transpose_1/kernel/Initializer/random_uniform/sub*
T0*6
_class,
*(loc:@generator/conv2d_transpose_1/kernel*&
_output_shapes
: @
?
>generator/conv2d_transpose_1/kernel/Initializer/random_uniformAddBgenerator/conv2d_transpose_1/kernel/Initializer/random_uniform/mulBgenerator/conv2d_transpose_1/kernel/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@generator/conv2d_transpose_1/kernel*&
_output_shapes
: @
?
#generator/conv2d_transpose_1/kernel
VariableV2*
dtype0*
shared_name *
shape: @*
	container *6
_class,
*(loc:@generator/conv2d_transpose_1/kernel*&
_output_shapes
: @
?
*generator/conv2d_transpose_1/kernel/AssignAssign#generator/conv2d_transpose_1/kernel>generator/conv2d_transpose_1/kernel/Initializer/random_uniform*
T0*
use_locking(*
validate_shape(*6
_class,
*(loc:@generator/conv2d_transpose_1/kernel*&
_output_shapes
: @
?
(generator/conv2d_transpose_1/kernel/readIdentity#generator/conv2d_transpose_1/kernel*
T0*6
_class,
*(loc:@generator/conv2d_transpose_1/kernel*&
_output_shapes
: @
?
3generator/conv2d_transpose_1/bias/Initializer/zerosConst*
valueB *    *
dtype0*4
_class*
(&loc:@generator/conv2d_transpose_1/bias*
_output_shapes
: 
?
!generator/conv2d_transpose_1/bias
VariableV2*
dtype0*
shared_name *
shape: *
	container *4
_class*
(&loc:@generator/conv2d_transpose_1/bias*
_output_shapes
: 
?
(generator/conv2d_transpose_1/bias/AssignAssign!generator/conv2d_transpose_1/bias3generator/conv2d_transpose_1/bias/Initializer/zeros*
T0*
use_locking(*
validate_shape(*4
_class*
(&loc:@generator/conv2d_transpose_1/bias*
_output_shapes
: 
?
&generator/conv2d_transpose_1/bias/readIdentity!generator/conv2d_transpose_1/bias*
T0*4
_class*
(&loc:@generator/conv2d_transpose_1/bias*
_output_shapes
: 
u
"generator/conv2d_transpose_1/ShapeShapegenerator/Maximum_1*
T0*
out_type0*
_output_shapes
:
z
0generator/conv2d_transpose_1/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
|
2generator/conv2d_transpose_1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
|
2generator/conv2d_transpose_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
*generator/conv2d_transpose_1/strided_sliceStridedSlice"generator/conv2d_transpose_1/Shape0generator/conv2d_transpose_1/strided_slice/stack2generator/conv2d_transpose_1/strided_slice/stack_12generator/conv2d_transpose_1/strided_slice/stack_2*
Index0*
end_mask *
shrink_axis_mask*
T0*

begin_mask *
new_axis_mask *
ellipsis_mask *
_output_shapes
: 
|
2generator/conv2d_transpose_1/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:
~
4generator/conv2d_transpose_1/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
~
4generator/conv2d_transpose_1/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
,generator/conv2d_transpose_1/strided_slice_1StridedSlice"generator/conv2d_transpose_1/Shape2generator/conv2d_transpose_1/strided_slice_1/stack4generator/conv2d_transpose_1/strided_slice_1/stack_14generator/conv2d_transpose_1/strided_slice_1/stack_2*
Index0*
end_mask *
shrink_axis_mask*
T0*

begin_mask *
new_axis_mask *
ellipsis_mask *
_output_shapes
: 
|
2generator/conv2d_transpose_1/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
~
4generator/conv2d_transpose_1/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
~
4generator/conv2d_transpose_1/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
,generator/conv2d_transpose_1/strided_slice_2StridedSlice"generator/conv2d_transpose_1/Shape2generator/conv2d_transpose_1/strided_slice_2/stack4generator/conv2d_transpose_1/strided_slice_2/stack_14generator/conv2d_transpose_1/strided_slice_2/stack_2*
Index0*
end_mask *
shrink_axis_mask*
T0*

begin_mask *
new_axis_mask *
ellipsis_mask *
_output_shapes
: 
d
"generator/conv2d_transpose_1/mul/yConst*
value	B :*
dtype0*
_output_shapes
: 
?
 generator/conv2d_transpose_1/mulMul,generator/conv2d_transpose_1/strided_slice_1"generator/conv2d_transpose_1/mul/y*
T0*
_output_shapes
: 
f
$generator/conv2d_transpose_1/mul_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
?
"generator/conv2d_transpose_1/mul_1Mul,generator/conv2d_transpose_1/strided_slice_2$generator/conv2d_transpose_1/mul_1/y*
T0*
_output_shapes
: 
f
$generator/conv2d_transpose_1/stack/3Const*
value	B : *
dtype0*
_output_shapes
: 
?
"generator/conv2d_transpose_1/stackPack*generator/conv2d_transpose_1/strided_slice generator/conv2d_transpose_1/mul"generator/conv2d_transpose_1/mul_1$generator/conv2d_transpose_1/stack/3*

axis *
T0*
N*
_output_shapes
:
?
-generator/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput"generator/conv2d_transpose_1/stack(generator/conv2d_transpose_1/kernel/readgenerator/Maximum_1*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(*/
_output_shapes
:????????? 
?
$generator/conv2d_transpose_1/BiasAddBiasAdd-generator/conv2d_transpose_1/conv2d_transpose&generator/conv2d_transpose_1/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:????????? 
?
6generator/batch_normalization_2/gamma/Initializer/onesConst*
valueB *  ??*
dtype0*8
_class.
,*loc:@generator/batch_normalization_2/gamma*
_output_shapes
: 
?
%generator/batch_normalization_2/gamma
VariableV2*
dtype0*
shared_name *
shape: *
	container *8
_class.
,*loc:@generator/batch_normalization_2/gamma*
_output_shapes
: 
?
,generator/batch_normalization_2/gamma/AssignAssign%generator/batch_normalization_2/gamma6generator/batch_normalization_2/gamma/Initializer/ones*
T0*
use_locking(*
validate_shape(*8
_class.
,*loc:@generator/batch_normalization_2/gamma*
_output_shapes
: 
?
*generator/batch_normalization_2/gamma/readIdentity%generator/batch_normalization_2/gamma*
T0*8
_class.
,*loc:@generator/batch_normalization_2/gamma*
_output_shapes
: 
?
6generator/batch_normalization_2/beta/Initializer/zerosConst*
valueB *    *
dtype0*7
_class-
+)loc:@generator/batch_normalization_2/beta*
_output_shapes
: 
?
$generator/batch_normalization_2/beta
VariableV2*
dtype0*
shared_name *
shape: *
	container *7
_class-
+)loc:@generator/batch_normalization_2/beta*
_output_shapes
: 
?
+generator/batch_normalization_2/beta/AssignAssign$generator/batch_normalization_2/beta6generator/batch_normalization_2/beta/Initializer/zeros*
T0*
use_locking(*
validate_shape(*7
_class-
+)loc:@generator/batch_normalization_2/beta*
_output_shapes
: 
?
)generator/batch_normalization_2/beta/readIdentity$generator/batch_normalization_2/beta*
T0*7
_class-
+)loc:@generator/batch_normalization_2/beta*
_output_shapes
: 
?
=generator/batch_normalization_2/moving_mean/Initializer/zerosConst*
valueB *    *
dtype0*>
_class4
20loc:@generator/batch_normalization_2/moving_mean*
_output_shapes
: 
?
+generator/batch_normalization_2/moving_mean
VariableV2*
dtype0*
shared_name *
shape: *
	container *>
_class4
20loc:@generator/batch_normalization_2/moving_mean*
_output_shapes
: 
?
2generator/batch_normalization_2/moving_mean/AssignAssign+generator/batch_normalization_2/moving_mean=generator/batch_normalization_2/moving_mean/Initializer/zeros*
T0*
use_locking(*
validate_shape(*>
_class4
20loc:@generator/batch_normalization_2/moving_mean*
_output_shapes
: 
?
0generator/batch_normalization_2/moving_mean/readIdentity+generator/batch_normalization_2/moving_mean*
T0*>
_class4
20loc:@generator/batch_normalization_2/moving_mean*
_output_shapes
: 
?
@generator/batch_normalization_2/moving_variance/Initializer/onesConst*
valueB *  ??*
dtype0*B
_class8
64loc:@generator/batch_normalization_2/moving_variance*
_output_shapes
: 
?
/generator/batch_normalization_2/moving_variance
VariableV2*
dtype0*
shared_name *
shape: *
	container *B
_class8
64loc:@generator/batch_normalization_2/moving_variance*
_output_shapes
: 
?
6generator/batch_normalization_2/moving_variance/AssignAssign/generator/batch_normalization_2/moving_variance@generator/batch_normalization_2/moving_variance/Initializer/ones*
T0*
use_locking(*
validate_shape(*B
_class8
64loc:@generator/batch_normalization_2/moving_variance*
_output_shapes
: 
?
4generator/batch_normalization_2/moving_variance/readIdentity/generator/batch_normalization_2/moving_variance*
T0*B
_class8
64loc:@generator/batch_normalization_2/moving_variance*
_output_shapes
: 
h
%generator/batch_normalization_2/ConstConst*
valueB *
dtype0*
_output_shapes
: 
j
'generator/batch_normalization_2/Const_1Const*
valueB *
dtype0*
_output_shapes
: 
?
.generator/batch_normalization_2/FusedBatchNormFusedBatchNorm$generator/conv2d_transpose_1/BiasAdd*generator/batch_normalization_2/gamma/read)generator/batch_normalization_2/beta/read%generator/batch_normalization_2/Const'generator/batch_normalization_2/Const_1*
is_training(*
T0*
data_formatNHWC*
epsilon%o?:*G
_output_shapes5
3:????????? : : : : 
l
'generator/batch_normalization_2/Const_2Const*
valueB
 *?p}?*
dtype0*
_output_shapes
: 
?
5generator/batch_normalization_2/AssignMovingAvg/sub/xConst*
valueB
 *  ??*
dtype0*>
_class4
20loc:@generator/batch_normalization_2/moving_mean*
_output_shapes
: 
?
3generator/batch_normalization_2/AssignMovingAvg/subSub5generator/batch_normalization_2/AssignMovingAvg/sub/x'generator/batch_normalization_2/Const_2*
T0*>
_class4
20loc:@generator/batch_normalization_2/moving_mean*
_output_shapes
: 
?
5generator/batch_normalization_2/AssignMovingAvg/sub_1Sub0generator/batch_normalization_2/moving_mean/read0generator/batch_normalization_2/FusedBatchNorm:1*
T0*>
_class4
20loc:@generator/batch_normalization_2/moving_mean*
_output_shapes
: 
?
3generator/batch_normalization_2/AssignMovingAvg/mulMul5generator/batch_normalization_2/AssignMovingAvg/sub_13generator/batch_normalization_2/AssignMovingAvg/sub*
T0*>
_class4
20loc:@generator/batch_normalization_2/moving_mean*
_output_shapes
: 
?
/generator/batch_normalization_2/AssignMovingAvg	AssignSub+generator/batch_normalization_2/moving_mean3generator/batch_normalization_2/AssignMovingAvg/mul*
T0*
use_locking( *>
_class4
20loc:@generator/batch_normalization_2/moving_mean*
_output_shapes
: 
?
7generator/batch_normalization_2/AssignMovingAvg_1/sub/xConst*
valueB
 *  ??*
dtype0*B
_class8
64loc:@generator/batch_normalization_2/moving_variance*
_output_shapes
: 
?
5generator/batch_normalization_2/AssignMovingAvg_1/subSub7generator/batch_normalization_2/AssignMovingAvg_1/sub/x'generator/batch_normalization_2/Const_2*
T0*B
_class8
64loc:@generator/batch_normalization_2/moving_variance*
_output_shapes
: 
?
7generator/batch_normalization_2/AssignMovingAvg_1/sub_1Sub4generator/batch_normalization_2/moving_variance/read0generator/batch_normalization_2/FusedBatchNorm:2*
T0*B
_class8
64loc:@generator/batch_normalization_2/moving_variance*
_output_shapes
: 
?
5generator/batch_normalization_2/AssignMovingAvg_1/mulMul7generator/batch_normalization_2/AssignMovingAvg_1/sub_15generator/batch_normalization_2/AssignMovingAvg_1/sub*
T0*B
_class8
64loc:@generator/batch_normalization_2/moving_variance*
_output_shapes
: 
?
1generator/batch_normalization_2/AssignMovingAvg_1	AssignSub/generator/batch_normalization_2/moving_variance5generator/batch_normalization_2/AssignMovingAvg_1/mul*
T0*
use_locking( *B
_class8
64loc:@generator/batch_normalization_2/moving_variance*
_output_shapes
: 
V
generator/mul_2/xConst*
valueB
 *??L>*
dtype0*
_output_shapes
: 
?
generator/mul_2Mulgenerator/mul_2/x.generator/batch_normalization_2/FusedBatchNorm*
T0*/
_output_shapes
:????????? 
?
generator/Maximum_2Maximumgenerator/mul_2.generator/batch_normalization_2/FusedBatchNorm*
T0*/
_output_shapes
:????????? 
?
Dgenerator/conv2d_transpose_2/kernel/Initializer/random_uniform/shapeConst*%
valueB"             *
dtype0*6
_class,
*(loc:@generator/conv2d_transpose_2/kernel*
_output_shapes
:
?
Bgenerator/conv2d_transpose_2/kernel/Initializer/random_uniform/minConst*
valueB
 *n???*
dtype0*6
_class,
*(loc:@generator/conv2d_transpose_2/kernel*
_output_shapes
: 
?
Bgenerator/conv2d_transpose_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *n??=*
dtype0*6
_class,
*(loc:@generator/conv2d_transpose_2/kernel*
_output_shapes
: 
?
Lgenerator/conv2d_transpose_2/kernel/Initializer/random_uniform/RandomUniformRandomUniformDgenerator/conv2d_transpose_2/kernel/Initializer/random_uniform/shape*
T0*
dtype0*
seed2 *

seed *6
_class,
*(loc:@generator/conv2d_transpose_2/kernel*&
_output_shapes
: 
?
Bgenerator/conv2d_transpose_2/kernel/Initializer/random_uniform/subSubBgenerator/conv2d_transpose_2/kernel/Initializer/random_uniform/maxBgenerator/conv2d_transpose_2/kernel/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@generator/conv2d_transpose_2/kernel*
_output_shapes
: 
?
Bgenerator/conv2d_transpose_2/kernel/Initializer/random_uniform/mulMulLgenerator/conv2d_transpose_2/kernel/Initializer/random_uniform/RandomUniformBgenerator/conv2d_transpose_2/kernel/Initializer/random_uniform/sub*
T0*6
_class,
*(loc:@generator/conv2d_transpose_2/kernel*&
_output_shapes
: 
?
>generator/conv2d_transpose_2/kernel/Initializer/random_uniformAddBgenerator/conv2d_transpose_2/kernel/Initializer/random_uniform/mulBgenerator/conv2d_transpose_2/kernel/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@generator/conv2d_transpose_2/kernel*&
_output_shapes
: 
?
#generator/conv2d_transpose_2/kernel
VariableV2*
dtype0*
shared_name *
shape: *
	container *6
_class,
*(loc:@generator/conv2d_transpose_2/kernel*&
_output_shapes
: 
?
*generator/conv2d_transpose_2/kernel/AssignAssign#generator/conv2d_transpose_2/kernel>generator/conv2d_transpose_2/kernel/Initializer/random_uniform*
T0*
use_locking(*
validate_shape(*6
_class,
*(loc:@generator/conv2d_transpose_2/kernel*&
_output_shapes
: 
?
(generator/conv2d_transpose_2/kernel/readIdentity#generator/conv2d_transpose_2/kernel*
T0*6
_class,
*(loc:@generator/conv2d_transpose_2/kernel*&
_output_shapes
: 
?
3generator/conv2d_transpose_2/bias/Initializer/zerosConst*
valueB*    *
dtype0*4
_class*
(&loc:@generator/conv2d_transpose_2/bias*
_output_shapes
:
?
!generator/conv2d_transpose_2/bias
VariableV2*
dtype0*
shared_name *
shape:*
	container *4
_class*
(&loc:@generator/conv2d_transpose_2/bias*
_output_shapes
:
?
(generator/conv2d_transpose_2/bias/AssignAssign!generator/conv2d_transpose_2/bias3generator/conv2d_transpose_2/bias/Initializer/zeros*
T0*
use_locking(*
validate_shape(*4
_class*
(&loc:@generator/conv2d_transpose_2/bias*
_output_shapes
:
?
&generator/conv2d_transpose_2/bias/readIdentity!generator/conv2d_transpose_2/bias*
T0*4
_class*
(&loc:@generator/conv2d_transpose_2/bias*
_output_shapes
:
u
"generator/conv2d_transpose_2/ShapeShapegenerator/Maximum_2*
T0*
out_type0*
_output_shapes
:
z
0generator/conv2d_transpose_2/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
|
2generator/conv2d_transpose_2/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
|
2generator/conv2d_transpose_2/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
*generator/conv2d_transpose_2/strided_sliceStridedSlice"generator/conv2d_transpose_2/Shape0generator/conv2d_transpose_2/strided_slice/stack2generator/conv2d_transpose_2/strided_slice/stack_12generator/conv2d_transpose_2/strided_slice/stack_2*
Index0*
end_mask *
shrink_axis_mask*
T0*

begin_mask *
new_axis_mask *
ellipsis_mask *
_output_shapes
: 
|
2generator/conv2d_transpose_2/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:
~
4generator/conv2d_transpose_2/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
~
4generator/conv2d_transpose_2/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
,generator/conv2d_transpose_2/strided_slice_1StridedSlice"generator/conv2d_transpose_2/Shape2generator/conv2d_transpose_2/strided_slice_1/stack4generator/conv2d_transpose_2/strided_slice_1/stack_14generator/conv2d_transpose_2/strided_slice_1/stack_2*
Index0*
end_mask *
shrink_axis_mask*
T0*

begin_mask *
new_axis_mask *
ellipsis_mask *
_output_shapes
: 
|
2generator/conv2d_transpose_2/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
~
4generator/conv2d_transpose_2/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
~
4generator/conv2d_transpose_2/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
,generator/conv2d_transpose_2/strided_slice_2StridedSlice"generator/conv2d_transpose_2/Shape2generator/conv2d_transpose_2/strided_slice_2/stack4generator/conv2d_transpose_2/strided_slice_2/stack_14generator/conv2d_transpose_2/strided_slice_2/stack_2*
Index0*
end_mask *
shrink_axis_mask*
T0*

begin_mask *
new_axis_mask *
ellipsis_mask *
_output_shapes
: 
d
"generator/conv2d_transpose_2/mul/yConst*
value	B :*
dtype0*
_output_shapes
: 
?
 generator/conv2d_transpose_2/mulMul,generator/conv2d_transpose_2/strided_slice_1"generator/conv2d_transpose_2/mul/y*
T0*
_output_shapes
: 
f
$generator/conv2d_transpose_2/mul_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
?
"generator/conv2d_transpose_2/mul_1Mul,generator/conv2d_transpose_2/strided_slice_2$generator/conv2d_transpose_2/mul_1/y*
T0*
_output_shapes
: 
f
$generator/conv2d_transpose_2/stack/3Const*
value	B :*
dtype0*
_output_shapes
: 
?
"generator/conv2d_transpose_2/stackPack*generator/conv2d_transpose_2/strided_slice generator/conv2d_transpose_2/mul"generator/conv2d_transpose_2/mul_1$generator/conv2d_transpose_2/stack/3*

axis *
T0*
N*
_output_shapes
:
?
-generator/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput"generator/conv2d_transpose_2/stack(generator/conv2d_transpose_2/kernel/readgenerator/Maximum_2*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(*/
_output_shapes
:?????????  
?
$generator/conv2d_transpose_2/BiasAddBiasAdd-generator/conv2d_transpose_2/conv2d_transpose&generator/conv2d_transpose_2/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:?????????  
v
generator/TanhTanh$generator/conv2d_transpose_2/BiasAdd*
T0*/
_output_shapes
:?????????  
\
discriminator/truediv/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
e
discriminator/truedivRealDiv	drop_ratediscriminator/truediv/y*
T0*
_output_shapes
: 
p
discriminator/dropout/IdentityIdentity
input_real*
T0*/
_output_shapes
:?????????  
?
<discriminator/conv2d/kernel/Initializer/random_uniform/shapeConst*%
valueB"         @   *
dtype0*.
_class$
" loc:@discriminator/conv2d/kernel*
_output_shapes
:
?
:discriminator/conv2d/kernel/Initializer/random_uniform/minConst*
valueB
 *?hϽ*
dtype0*.
_class$
" loc:@discriminator/conv2d/kernel*
_output_shapes
: 
?
:discriminator/conv2d/kernel/Initializer/random_uniform/maxConst*
valueB
 *?h?=*
dtype0*.
_class$
" loc:@discriminator/conv2d/kernel*
_output_shapes
: 
?
Ddiscriminator/conv2d/kernel/Initializer/random_uniform/RandomUniformRandomUniform<discriminator/conv2d/kernel/Initializer/random_uniform/shape*
T0*
dtype0*
seed2 *

seed *.
_class$
" loc:@discriminator/conv2d/kernel*&
_output_shapes
:@
?
:discriminator/conv2d/kernel/Initializer/random_uniform/subSub:discriminator/conv2d/kernel/Initializer/random_uniform/max:discriminator/conv2d/kernel/Initializer/random_uniform/min*
T0*.
_class$
" loc:@discriminator/conv2d/kernel*
_output_shapes
: 
?
:discriminator/conv2d/kernel/Initializer/random_uniform/mulMulDdiscriminator/conv2d/kernel/Initializer/random_uniform/RandomUniform:discriminator/conv2d/kernel/Initializer/random_uniform/sub*
T0*.
_class$
" loc:@discriminator/conv2d/kernel*&
_output_shapes
:@
?
6discriminator/conv2d/kernel/Initializer/random_uniformAdd:discriminator/conv2d/kernel/Initializer/random_uniform/mul:discriminator/conv2d/kernel/Initializer/random_uniform/min*
T0*.
_class$
" loc:@discriminator/conv2d/kernel*&
_output_shapes
:@
?
discriminator/conv2d/kernel
VariableV2*
dtype0*
shared_name *
shape:@*
	container *.
_class$
" loc:@discriminator/conv2d/kernel*&
_output_shapes
:@
?
"discriminator/conv2d/kernel/AssignAssigndiscriminator/conv2d/kernel6discriminator/conv2d/kernel/Initializer/random_uniform*
T0*
use_locking(*
validate_shape(*.
_class$
" loc:@discriminator/conv2d/kernel*&
_output_shapes
:@
?
 discriminator/conv2d/kernel/readIdentitydiscriminator/conv2d/kernel*
T0*.
_class$
" loc:@discriminator/conv2d/kernel*&
_output_shapes
:@
?
+discriminator/conv2d/bias/Initializer/zerosConst*
valueB@*    *
dtype0*,
_class"
 loc:@discriminator/conv2d/bias*
_output_shapes
:@
?
discriminator/conv2d/bias
VariableV2*
dtype0*
shared_name *
shape:@*
	container *,
_class"
 loc:@discriminator/conv2d/bias*
_output_shapes
:@
?
 discriminator/conv2d/bias/AssignAssigndiscriminator/conv2d/bias+discriminator/conv2d/bias/Initializer/zeros*
T0*
use_locking(*
validate_shape(*,
_class"
 loc:@discriminator/conv2d/bias*
_output_shapes
:@
?
discriminator/conv2d/bias/readIdentitydiscriminator/conv2d/bias*
T0*,
_class"
 loc:@discriminator/conv2d/bias*
_output_shapes
:@
s
"discriminator/conv2d/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
discriminator/conv2d/Conv2DConv2Ddiscriminator/dropout/Identity discriminator/conv2d/kernel/read*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(*/
_output_shapes
:?????????@
?
discriminator/conv2d/BiasAddBiasAdddiscriminator/conv2d/Conv2Ddiscriminator/conv2d/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:?????????@
y
discriminator/conv2d/ReluReludiscriminator/conv2d/BiasAdd*
T0*/
_output_shapes
:?????????@
?
 discriminator/dropout_1/IdentityIdentitydiscriminator/conv2d/Relu*
T0*/
_output_shapes
:?????????@
?
>discriminator/conv2d_1/kernel/Initializer/random_uniform/shapeConst*%
valueB"      @   @   *
dtype0*0
_class&
$"loc:@discriminator/conv2d_1/kernel*
_output_shapes
:
?
<discriminator/conv2d_1/kernel/Initializer/random_uniform/minConst*
valueB
 *:͓?*
dtype0*0
_class&
$"loc:@discriminator/conv2d_1/kernel*
_output_shapes
: 
?
<discriminator/conv2d_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *:͓=*
dtype0*0
_class&
$"loc:@discriminator/conv2d_1/kernel*
_output_shapes
: 
?
Fdiscriminator/conv2d_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform>discriminator/conv2d_1/kernel/Initializer/random_uniform/shape*
T0*
dtype0*
seed2 *

seed *0
_class&
$"loc:@discriminator/conv2d_1/kernel*&
_output_shapes
:@@
?
<discriminator/conv2d_1/kernel/Initializer/random_uniform/subSub<discriminator/conv2d_1/kernel/Initializer/random_uniform/max<discriminator/conv2d_1/kernel/Initializer/random_uniform/min*
T0*0
_class&
$"loc:@discriminator/conv2d_1/kernel*
_output_shapes
: 
?
<discriminator/conv2d_1/kernel/Initializer/random_uniform/mulMulFdiscriminator/conv2d_1/kernel/Initializer/random_uniform/RandomUniform<discriminator/conv2d_1/kernel/Initializer/random_uniform/sub*
T0*0
_class&
$"loc:@discriminator/conv2d_1/kernel*&
_output_shapes
:@@
?
8discriminator/conv2d_1/kernel/Initializer/random_uniformAdd<discriminator/conv2d_1/kernel/Initializer/random_uniform/mul<discriminator/conv2d_1/kernel/Initializer/random_uniform/min*
T0*0
_class&
$"loc:@discriminator/conv2d_1/kernel*&
_output_shapes
:@@
?
discriminator/conv2d_1/kernel
VariableV2*
dtype0*
shared_name *
shape:@@*
	container *0
_class&
$"loc:@discriminator/conv2d_1/kernel*&
_output_shapes
:@@
?
$discriminator/conv2d_1/kernel/AssignAssigndiscriminator/conv2d_1/kernel8discriminator/conv2d_1/kernel/Initializer/random_uniform*
T0*
use_locking(*
validate_shape(*0
_class&
$"loc:@discriminator/conv2d_1/kernel*&
_output_shapes
:@@
?
"discriminator/conv2d_1/kernel/readIdentitydiscriminator/conv2d_1/kernel*
T0*0
_class&
$"loc:@discriminator/conv2d_1/kernel*&
_output_shapes
:@@
?
-discriminator/conv2d_1/bias/Initializer/zerosConst*
valueB@*    *
dtype0*.
_class$
" loc:@discriminator/conv2d_1/bias*
_output_shapes
:@
?
discriminator/conv2d_1/bias
VariableV2*
dtype0*
shared_name *
shape:@*
	container *.
_class$
" loc:@discriminator/conv2d_1/bias*
_output_shapes
:@
?
"discriminator/conv2d_1/bias/AssignAssigndiscriminator/conv2d_1/bias-discriminator/conv2d_1/bias/Initializer/zeros*
T0*
use_locking(*
validate_shape(*.
_class$
" loc:@discriminator/conv2d_1/bias*
_output_shapes
:@
?
 discriminator/conv2d_1/bias/readIdentitydiscriminator/conv2d_1/bias*
T0*.
_class$
" loc:@discriminator/conv2d_1/bias*
_output_shapes
:@
u
$discriminator/conv2d_1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
discriminator/conv2d_1/Conv2DConv2D discriminator/dropout_1/Identity"discriminator/conv2d_1/kernel/read*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(*/
_output_shapes
:?????????@
?
discriminator/conv2d_1/BiasAddBiasAdddiscriminator/conv2d_1/Conv2D discriminator/conv2d_1/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:?????????@
}
discriminator/conv2d_1/ReluReludiscriminator/conv2d_1/BiasAdd*
T0*/
_output_shapes
:?????????@
?
>discriminator/conv2d_2/kernel/Initializer/random_uniform/shapeConst*%
valueB"      @   @   *
dtype0*0
_class&
$"loc:@discriminator/conv2d_2/kernel*
_output_shapes
:
?
<discriminator/conv2d_2/kernel/Initializer/random_uniform/minConst*
valueB
 *:͓?*
dtype0*0
_class&
$"loc:@discriminator/conv2d_2/kernel*
_output_shapes
: 
?
<discriminator/conv2d_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *:͓=*
dtype0*0
_class&
$"loc:@discriminator/conv2d_2/kernel*
_output_shapes
: 
?
Fdiscriminator/conv2d_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform>discriminator/conv2d_2/kernel/Initializer/random_uniform/shape*
T0*
dtype0*
seed2 *

seed *0
_class&
$"loc:@discriminator/conv2d_2/kernel*&
_output_shapes
:@@
?
<discriminator/conv2d_2/kernel/Initializer/random_uniform/subSub<discriminator/conv2d_2/kernel/Initializer/random_uniform/max<discriminator/conv2d_2/kernel/Initializer/random_uniform/min*
T0*0
_class&
$"loc:@discriminator/conv2d_2/kernel*
_output_shapes
: 
?
<discriminator/conv2d_2/kernel/Initializer/random_uniform/mulMulFdiscriminator/conv2d_2/kernel/Initializer/random_uniform/RandomUniform<discriminator/conv2d_2/kernel/Initializer/random_uniform/sub*
T0*0
_class&
$"loc:@discriminator/conv2d_2/kernel*&
_output_shapes
:@@
?
8discriminator/conv2d_2/kernel/Initializer/random_uniformAdd<discriminator/conv2d_2/kernel/Initializer/random_uniform/mul<discriminator/conv2d_2/kernel/Initializer/random_uniform/min*
T0*0
_class&
$"loc:@discriminator/conv2d_2/kernel*&
_output_shapes
:@@
?
discriminator/conv2d_2/kernel
VariableV2*
dtype0*
shared_name *
shape:@@*
	container *0
_class&
$"loc:@discriminator/conv2d_2/kernel*&
_output_shapes
:@@
?
$discriminator/conv2d_2/kernel/AssignAssigndiscriminator/conv2d_2/kernel8discriminator/conv2d_2/kernel/Initializer/random_uniform*
T0*
use_locking(*
validate_shape(*0
_class&
$"loc:@discriminator/conv2d_2/kernel*&
_output_shapes
:@@
?
"discriminator/conv2d_2/kernel/readIdentitydiscriminator/conv2d_2/kernel*
T0*0
_class&
$"loc:@discriminator/conv2d_2/kernel*&
_output_shapes
:@@
?
-discriminator/conv2d_2/bias/Initializer/zerosConst*
valueB@*    *
dtype0*.
_class$
" loc:@discriminator/conv2d_2/bias*
_output_shapes
:@
?
discriminator/conv2d_2/bias
VariableV2*
dtype0*
shared_name *
shape:@*
	container *.
_class$
" loc:@discriminator/conv2d_2/bias*
_output_shapes
:@
?
"discriminator/conv2d_2/bias/AssignAssigndiscriminator/conv2d_2/bias-discriminator/conv2d_2/bias/Initializer/zeros*
T0*
use_locking(*
validate_shape(*.
_class$
" loc:@discriminator/conv2d_2/bias*
_output_shapes
:@
?
 discriminator/conv2d_2/bias/readIdentitydiscriminator/conv2d_2/bias*
T0*.
_class$
" loc:@discriminator/conv2d_2/bias*
_output_shapes
:@
u
$discriminator/conv2d_2/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
discriminator/conv2d_2/Conv2DConv2Ddiscriminator/conv2d_1/Relu"discriminator/conv2d_2/kernel/read*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(*/
_output_shapes
:?????????@
?
discriminator/conv2d_2/BiasAddBiasAdddiscriminator/conv2d_2/Conv2D discriminator/conv2d_2/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:?????????@
?
8discriminator/batch_normalization/gamma/Initializer/onesConst*
valueB@*  ??*
dtype0*:
_class0
.,loc:@discriminator/batch_normalization/gamma*
_output_shapes
:@
?
'discriminator/batch_normalization/gamma
VariableV2*
dtype0*
shared_name *
shape:@*
	container *:
_class0
.,loc:@discriminator/batch_normalization/gamma*
_output_shapes
:@
?
.discriminator/batch_normalization/gamma/AssignAssign'discriminator/batch_normalization/gamma8discriminator/batch_normalization/gamma/Initializer/ones*
T0*
use_locking(*
validate_shape(*:
_class0
.,loc:@discriminator/batch_normalization/gamma*
_output_shapes
:@
?
,discriminator/batch_normalization/gamma/readIdentity'discriminator/batch_normalization/gamma*
T0*:
_class0
.,loc:@discriminator/batch_normalization/gamma*
_output_shapes
:@
?
8discriminator/batch_normalization/beta/Initializer/zerosConst*
valueB@*    *
dtype0*9
_class/
-+loc:@discriminator/batch_normalization/beta*
_output_shapes
:@
?
&discriminator/batch_normalization/beta
VariableV2*
dtype0*
shared_name *
shape:@*
	container *9
_class/
-+loc:@discriminator/batch_normalization/beta*
_output_shapes
:@
?
-discriminator/batch_normalization/beta/AssignAssign&discriminator/batch_normalization/beta8discriminator/batch_normalization/beta/Initializer/zeros*
T0*
use_locking(*
validate_shape(*9
_class/
-+loc:@discriminator/batch_normalization/beta*
_output_shapes
:@
?
+discriminator/batch_normalization/beta/readIdentity&discriminator/batch_normalization/beta*
T0*9
_class/
-+loc:@discriminator/batch_normalization/beta*
_output_shapes
:@
?
?discriminator/batch_normalization/moving_mean/Initializer/zerosConst*
valueB@*    *
dtype0*@
_class6
42loc:@discriminator/batch_normalization/moving_mean*
_output_shapes
:@
?
-discriminator/batch_normalization/moving_mean
VariableV2*
dtype0*
shared_name *
shape:@*
	container *@
_class6
42loc:@discriminator/batch_normalization/moving_mean*
_output_shapes
:@
?
4discriminator/batch_normalization/moving_mean/AssignAssign-discriminator/batch_normalization/moving_mean?discriminator/batch_normalization/moving_mean/Initializer/zeros*
T0*
use_locking(*
validate_shape(*@
_class6
42loc:@discriminator/batch_normalization/moving_mean*
_output_shapes
:@
?
2discriminator/batch_normalization/moving_mean/readIdentity-discriminator/batch_normalization/moving_mean*
T0*@
_class6
42loc:@discriminator/batch_normalization/moving_mean*
_output_shapes
:@
?
Bdiscriminator/batch_normalization/moving_variance/Initializer/onesConst*
valueB@*  ??*
dtype0*D
_class:
86loc:@discriminator/batch_normalization/moving_variance*
_output_shapes
:@
?
1discriminator/batch_normalization/moving_variance
VariableV2*
dtype0*
shared_name *
shape:@*
	container *D
_class:
86loc:@discriminator/batch_normalization/moving_variance*
_output_shapes
:@
?
8discriminator/batch_normalization/moving_variance/AssignAssign1discriminator/batch_normalization/moving_varianceBdiscriminator/batch_normalization/moving_variance/Initializer/ones*
T0*
use_locking(*
validate_shape(*D
_class:
86loc:@discriminator/batch_normalization/moving_variance*
_output_shapes
:@
?
6discriminator/batch_normalization/moving_variance/readIdentity1discriminator/batch_normalization/moving_variance*
T0*D
_class:
86loc:@discriminator/batch_normalization/moving_variance*
_output_shapes
:@
?
0discriminator/batch_normalization/FusedBatchNormFusedBatchNormdiscriminator/conv2d_2/BiasAdd,discriminator/batch_normalization/gamma/read+discriminator/batch_normalization/beta/read2discriminator/batch_normalization/moving_mean/read6discriminator/batch_normalization/moving_variance/read*
is_training( *
T0*
data_formatNHWC*
epsilon%o?:*G
_output_shapes5
3:?????????@:@:@:@:@
l
'discriminator/batch_normalization/ConstConst*
valueB
 *?p}?*
dtype0*
_output_shapes
: 
?
discriminator/ReluRelu0discriminator/batch_normalization/FusedBatchNorm*
T0*/
_output_shapes
:?????????@
z
 discriminator/dropout_2/IdentityIdentitydiscriminator/Relu*
T0*/
_output_shapes
:?????????@
?
>discriminator/conv2d_3/kernel/Initializer/random_uniform/shapeConst*%
valueB"      @   ?   *
dtype0*0
_class&
$"loc:@discriminator/conv2d_3/kernel*
_output_shapes
:
?
<discriminator/conv2d_3/kernel/Initializer/random_uniform/minConst*
valueB
 *?[q?*
dtype0*0
_class&
$"loc:@discriminator/conv2d_3/kernel*
_output_shapes
: 
?
<discriminator/conv2d_3/kernel/Initializer/random_uniform/maxConst*
valueB
 *?[q=*
dtype0*0
_class&
$"loc:@discriminator/conv2d_3/kernel*
_output_shapes
: 
?
Fdiscriminator/conv2d_3/kernel/Initializer/random_uniform/RandomUniformRandomUniform>discriminator/conv2d_3/kernel/Initializer/random_uniform/shape*
T0*
dtype0*
seed2 *

seed *0
_class&
$"loc:@discriminator/conv2d_3/kernel*'
_output_shapes
:@?
?
<discriminator/conv2d_3/kernel/Initializer/random_uniform/subSub<discriminator/conv2d_3/kernel/Initializer/random_uniform/max<discriminator/conv2d_3/kernel/Initializer/random_uniform/min*
T0*0
_class&
$"loc:@discriminator/conv2d_3/kernel*
_output_shapes
: 
?
<discriminator/conv2d_3/kernel/Initializer/random_uniform/mulMulFdiscriminator/conv2d_3/kernel/Initializer/random_uniform/RandomUniform<discriminator/conv2d_3/kernel/Initializer/random_uniform/sub*
T0*0
_class&
$"loc:@discriminator/conv2d_3/kernel*'
_output_shapes
:@?
?
8discriminator/conv2d_3/kernel/Initializer/random_uniformAdd<discriminator/conv2d_3/kernel/Initializer/random_uniform/mul<discriminator/conv2d_3/kernel/Initializer/random_uniform/min*
T0*0
_class&
$"loc:@discriminator/conv2d_3/kernel*'
_output_shapes
:@?
?
discriminator/conv2d_3/kernel
VariableV2*
dtype0*
shared_name *
shape:@?*
	container *0
_class&
$"loc:@discriminator/conv2d_3/kernel*'
_output_shapes
:@?
?
$discriminator/conv2d_3/kernel/AssignAssigndiscriminator/conv2d_3/kernel8discriminator/conv2d_3/kernel/Initializer/random_uniform*
T0*
use_locking(*
validate_shape(*0
_class&
$"loc:@discriminator/conv2d_3/kernel*'
_output_shapes
:@?
?
"discriminator/conv2d_3/kernel/readIdentitydiscriminator/conv2d_3/kernel*
T0*0
_class&
$"loc:@discriminator/conv2d_3/kernel*'
_output_shapes
:@?
?
-discriminator/conv2d_3/bias/Initializer/zerosConst*
valueB?*    *
dtype0*.
_class$
" loc:@discriminator/conv2d_3/bias*
_output_shapes	
:?
?
discriminator/conv2d_3/bias
VariableV2*
dtype0*
shared_name *
shape:?*
	container *.
_class$
" loc:@discriminator/conv2d_3/bias*
_output_shapes	
:?
?
"discriminator/conv2d_3/bias/AssignAssigndiscriminator/conv2d_3/bias-discriminator/conv2d_3/bias/Initializer/zeros*
T0*
use_locking(*
validate_shape(*.
_class$
" loc:@discriminator/conv2d_3/bias*
_output_shapes	
:?
?
 discriminator/conv2d_3/bias/readIdentitydiscriminator/conv2d_3/bias*
T0*.
_class$
" loc:@discriminator/conv2d_3/bias*
_output_shapes	
:?
u
$discriminator/conv2d_3/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
discriminator/conv2d_3/Conv2DConv2D discriminator/dropout_2/Identity"discriminator/conv2d_3/kernel/read*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(*0
_output_shapes
:??????????
?
discriminator/conv2d_3/BiasAddBiasAdddiscriminator/conv2d_3/Conv2D discriminator/conv2d_3/bias/read*
T0*
data_formatNHWC*0
_output_shapes
:??????????
?
:discriminator/batch_normalization_1/gamma/Initializer/onesConst*
valueB?*  ??*
dtype0*<
_class2
0.loc:@discriminator/batch_normalization_1/gamma*
_output_shapes	
:?
?
)discriminator/batch_normalization_1/gamma
VariableV2*
dtype0*
shared_name *
shape:?*
	container *<
_class2
0.loc:@discriminator/batch_normalization_1/gamma*
_output_shapes	
:?
?
0discriminator/batch_normalization_1/gamma/AssignAssign)discriminator/batch_normalization_1/gamma:discriminator/batch_normalization_1/gamma/Initializer/ones*
T0*
use_locking(*
validate_shape(*<
_class2
0.loc:@discriminator/batch_normalization_1/gamma*
_output_shapes	
:?
?
.discriminator/batch_normalization_1/gamma/readIdentity)discriminator/batch_normalization_1/gamma*
T0*<
_class2
0.loc:@discriminator/batch_normalization_1/gamma*
_output_shapes	
:?
?
:discriminator/batch_normalization_1/beta/Initializer/zerosConst*
valueB?*    *
dtype0*;
_class1
/-loc:@discriminator/batch_normalization_1/beta*
_output_shapes	
:?
?
(discriminator/batch_normalization_1/beta
VariableV2*
dtype0*
shared_name *
shape:?*
	container *;
_class1
/-loc:@discriminator/batch_normalization_1/beta*
_output_shapes	
:?
?
/discriminator/batch_normalization_1/beta/AssignAssign(discriminator/batch_normalization_1/beta:discriminator/batch_normalization_1/beta/Initializer/zeros*
T0*
use_locking(*
validate_shape(*;
_class1
/-loc:@discriminator/batch_normalization_1/beta*
_output_shapes	
:?
?
-discriminator/batch_normalization_1/beta/readIdentity(discriminator/batch_normalization_1/beta*
T0*;
_class1
/-loc:@discriminator/batch_normalization_1/beta*
_output_shapes	
:?
?
Adiscriminator/batch_normalization_1/moving_mean/Initializer/zerosConst*
valueB?*    *
dtype0*B
_class8
64loc:@discriminator/batch_normalization_1/moving_mean*
_output_shapes	
:?
?
/discriminator/batch_normalization_1/moving_mean
VariableV2*
dtype0*
shared_name *
shape:?*
	container *B
_class8
64loc:@discriminator/batch_normalization_1/moving_mean*
_output_shapes	
:?
?
6discriminator/batch_normalization_1/moving_mean/AssignAssign/discriminator/batch_normalization_1/moving_meanAdiscriminator/batch_normalization_1/moving_mean/Initializer/zeros*
T0*
use_locking(*
validate_shape(*B
_class8
64loc:@discriminator/batch_normalization_1/moving_mean*
_output_shapes	
:?
?
4discriminator/batch_normalization_1/moving_mean/readIdentity/discriminator/batch_normalization_1/moving_mean*
T0*B
_class8
64loc:@discriminator/batch_normalization_1/moving_mean*
_output_shapes	
:?
?
Ddiscriminator/batch_normalization_1/moving_variance/Initializer/onesConst*
valueB?*  ??*
dtype0*F
_class<
:8loc:@discriminator/batch_normalization_1/moving_variance*
_output_shapes	
:?
?
3discriminator/batch_normalization_1/moving_variance
VariableV2*
dtype0*
shared_name *
shape:?*
	container *F
_class<
:8loc:@discriminator/batch_normalization_1/moving_variance*
_output_shapes	
:?
?
:discriminator/batch_normalization_1/moving_variance/AssignAssign3discriminator/batch_normalization_1/moving_varianceDdiscriminator/batch_normalization_1/moving_variance/Initializer/ones*
T0*
use_locking(*
validate_shape(*F
_class<
:8loc:@discriminator/batch_normalization_1/moving_variance*
_output_shapes	
:?
?
8discriminator/batch_normalization_1/moving_variance/readIdentity3discriminator/batch_normalization_1/moving_variance*
T0*F
_class<
:8loc:@discriminator/batch_normalization_1/moving_variance*
_output_shapes	
:?
?
2discriminator/batch_normalization_1/FusedBatchNormFusedBatchNormdiscriminator/conv2d_3/BiasAdd.discriminator/batch_normalization_1/gamma/read-discriminator/batch_normalization_1/beta/read4discriminator/batch_normalization_1/moving_mean/read8discriminator/batch_normalization_1/moving_variance/read*
is_training( *
T0*
data_formatNHWC*
epsilon%o?:*L
_output_shapes:
8:??????????:?:?:?:?
n
)discriminator/batch_normalization_1/ConstConst*
valueB
 *?p}?*
dtype0*
_output_shapes
: 
?
discriminator/Relu_1Relu2discriminator/batch_normalization_1/FusedBatchNorm*
T0*0
_output_shapes
:??????????
?
>discriminator/conv2d_4/kernel/Initializer/random_uniform/shapeConst*%
valueB"      ?   ?   *
dtype0*0
_class&
$"loc:@discriminator/conv2d_4/kernel*
_output_shapes
:
?
<discriminator/conv2d_4/kernel/Initializer/random_uniform/minConst*
valueB
 *?Q?*
dtype0*0
_class&
$"loc:@discriminator/conv2d_4/kernel*
_output_shapes
: 
?
<discriminator/conv2d_4/kernel/Initializer/random_uniform/maxConst*
valueB
 *?Q=*
dtype0*0
_class&
$"loc:@discriminator/conv2d_4/kernel*
_output_shapes
: 
?
Fdiscriminator/conv2d_4/kernel/Initializer/random_uniform/RandomUniformRandomUniform>discriminator/conv2d_4/kernel/Initializer/random_uniform/shape*
T0*
dtype0*
seed2 *

seed *0
_class&
$"loc:@discriminator/conv2d_4/kernel*(
_output_shapes
:??
?
<discriminator/conv2d_4/kernel/Initializer/random_uniform/subSub<discriminator/conv2d_4/kernel/Initializer/random_uniform/max<discriminator/conv2d_4/kernel/Initializer/random_uniform/min*
T0*0
_class&
$"loc:@discriminator/conv2d_4/kernel*
_output_shapes
: 
?
<discriminator/conv2d_4/kernel/Initializer/random_uniform/mulMulFdiscriminator/conv2d_4/kernel/Initializer/random_uniform/RandomUniform<discriminator/conv2d_4/kernel/Initializer/random_uniform/sub*
T0*0
_class&
$"loc:@discriminator/conv2d_4/kernel*(
_output_shapes
:??
?
8discriminator/conv2d_4/kernel/Initializer/random_uniformAdd<discriminator/conv2d_4/kernel/Initializer/random_uniform/mul<discriminator/conv2d_4/kernel/Initializer/random_uniform/min*
T0*0
_class&
$"loc:@discriminator/conv2d_4/kernel*(
_output_shapes
:??
?
discriminator/conv2d_4/kernel
VariableV2*
dtype0*
shared_name *
shape:??*
	container *0
_class&
$"loc:@discriminator/conv2d_4/kernel*(
_output_shapes
:??
?
$discriminator/conv2d_4/kernel/AssignAssigndiscriminator/conv2d_4/kernel8discriminator/conv2d_4/kernel/Initializer/random_uniform*
T0*
use_locking(*
validate_shape(*0
_class&
$"loc:@discriminator/conv2d_4/kernel*(
_output_shapes
:??
?
"discriminator/conv2d_4/kernel/readIdentitydiscriminator/conv2d_4/kernel*
T0*0
_class&
$"loc:@discriminator/conv2d_4/kernel*(
_output_shapes
:??
?
-discriminator/conv2d_4/bias/Initializer/zerosConst*
valueB?*    *
dtype0*.
_class$
" loc:@discriminator/conv2d_4/bias*
_output_shapes	
:?
?
discriminator/conv2d_4/bias
VariableV2*
dtype0*
shared_name *
shape:?*
	container *.
_class$
" loc:@discriminator/conv2d_4/bias*
_output_shapes	
:?
?
"discriminator/conv2d_4/bias/AssignAssigndiscriminator/conv2d_4/bias-discriminator/conv2d_4/bias/Initializer/zeros*
T0*
use_locking(*
validate_shape(*.
_class$
" loc:@discriminator/conv2d_4/bias*
_output_shapes	
:?
?
 discriminator/conv2d_4/bias/readIdentitydiscriminator/conv2d_4/bias*
T0*.
_class$
" loc:@discriminator/conv2d_4/bias*
_output_shapes	
:?
u
$discriminator/conv2d_4/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
discriminator/conv2d_4/Conv2DConv2Ddiscriminator/Relu_1"discriminator/conv2d_4/kernel/read*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(*0
_output_shapes
:??????????
?
discriminator/conv2d_4/BiasAddBiasAdddiscriminator/conv2d_4/Conv2D discriminator/conv2d_4/bias/read*
T0*
data_formatNHWC*0
_output_shapes
:??????????
?
:discriminator/batch_normalization_2/gamma/Initializer/onesConst*
valueB?*  ??*
dtype0*<
_class2
0.loc:@discriminator/batch_normalization_2/gamma*
_output_shapes	
:?
?
)discriminator/batch_normalization_2/gamma
VariableV2*
dtype0*
shared_name *
shape:?*
	container *<
_class2
0.loc:@discriminator/batch_normalization_2/gamma*
_output_shapes	
:?
?
0discriminator/batch_normalization_2/gamma/AssignAssign)discriminator/batch_normalization_2/gamma:discriminator/batch_normalization_2/gamma/Initializer/ones*
T0*
use_locking(*
validate_shape(*<
_class2
0.loc:@discriminator/batch_normalization_2/gamma*
_output_shapes	
:?
?
.discriminator/batch_normalization_2/gamma/readIdentity)discriminator/batch_normalization_2/gamma*
T0*<
_class2
0.loc:@discriminator/batch_normalization_2/gamma*
_output_shapes	
:?
?
:discriminator/batch_normalization_2/beta/Initializer/zerosConst*
valueB?*    *
dtype0*;
_class1
/-loc:@discriminator/batch_normalization_2/beta*
_output_shapes	
:?
?
(discriminator/batch_normalization_2/beta
VariableV2*
dtype0*
shared_name *
shape:?*
	container *;
_class1
/-loc:@discriminator/batch_normalization_2/beta*
_output_shapes	
:?
?
/discriminator/batch_normalization_2/beta/AssignAssign(discriminator/batch_normalization_2/beta:discriminator/batch_normalization_2/beta/Initializer/zeros*
T0*
use_locking(*
validate_shape(*;
_class1
/-loc:@discriminator/batch_normalization_2/beta*
_output_shapes	
:?
?
-discriminator/batch_normalization_2/beta/readIdentity(discriminator/batch_normalization_2/beta*
T0*;
_class1
/-loc:@discriminator/batch_normalization_2/beta*
_output_shapes	
:?
?
Adiscriminator/batch_normalization_2/moving_mean/Initializer/zerosConst*
valueB?*    *
dtype0*B
_class8
64loc:@discriminator/batch_normalization_2/moving_mean*
_output_shapes	
:?
?
/discriminator/batch_normalization_2/moving_mean
VariableV2*
dtype0*
shared_name *
shape:?*
	container *B
_class8
64loc:@discriminator/batch_normalization_2/moving_mean*
_output_shapes	
:?
?
6discriminator/batch_normalization_2/moving_mean/AssignAssign/discriminator/batch_normalization_2/moving_meanAdiscriminator/batch_normalization_2/moving_mean/Initializer/zeros*
T0*
use_locking(*
validate_shape(*B
_class8
64loc:@discriminator/batch_normalization_2/moving_mean*
_output_shapes	
:?
?
4discriminator/batch_normalization_2/moving_mean/readIdentity/discriminator/batch_normalization_2/moving_mean*
T0*B
_class8
64loc:@discriminator/batch_normalization_2/moving_mean*
_output_shapes	
:?
?
Ddiscriminator/batch_normalization_2/moving_variance/Initializer/onesConst*
valueB?*  ??*
dtype0*F
_class<
:8loc:@discriminator/batch_normalization_2/moving_variance*
_output_shapes	
:?
?
3discriminator/batch_normalization_2/moving_variance
VariableV2*
dtype0*
shared_name *
shape:?*
	container *F
_class<
:8loc:@discriminator/batch_normalization_2/moving_variance*
_output_shapes	
:?
?
:discriminator/batch_normalization_2/moving_variance/AssignAssign3discriminator/batch_normalization_2/moving_varianceDdiscriminator/batch_normalization_2/moving_variance/Initializer/ones*
T0*
use_locking(*
validate_shape(*F
_class<
:8loc:@discriminator/batch_normalization_2/moving_variance*
_output_shapes	
:?
?
8discriminator/batch_normalization_2/moving_variance/readIdentity3discriminator/batch_normalization_2/moving_variance*
T0*F
_class<
:8loc:@discriminator/batch_normalization_2/moving_variance*
_output_shapes	
:?
?
2discriminator/batch_normalization_2/FusedBatchNormFusedBatchNormdiscriminator/conv2d_4/BiasAdd.discriminator/batch_normalization_2/gamma/read-discriminator/batch_normalization_2/beta/read4discriminator/batch_normalization_2/moving_mean/read8discriminator/batch_normalization_2/moving_variance/read*
is_training( *
T0*
data_formatNHWC*
epsilon%o?:*L
_output_shapes:
8:??????????:?:?:?:?
n
)discriminator/batch_normalization_2/ConstConst*
valueB
 *?p}?*
dtype0*
_output_shapes
: 
?
discriminator/Relu_2Relu2discriminator/batch_normalization_2/FusedBatchNorm*
T0*0
_output_shapes
:??????????
?
>discriminator/conv2d_5/kernel/Initializer/random_uniform/shapeConst*%
valueB"      ?   ?   *
dtype0*0
_class&
$"loc:@discriminator/conv2d_5/kernel*
_output_shapes
:
?
<discriminator/conv2d_5/kernel/Initializer/random_uniform/minConst*
valueB
 *?Q?*
dtype0*0
_class&
$"loc:@discriminator/conv2d_5/kernel*
_output_shapes
: 
?
<discriminator/conv2d_5/kernel/Initializer/random_uniform/maxConst*
valueB
 *?Q=*
dtype0*0
_class&
$"loc:@discriminator/conv2d_5/kernel*
_output_shapes
: 
?
Fdiscriminator/conv2d_5/kernel/Initializer/random_uniform/RandomUniformRandomUniform>discriminator/conv2d_5/kernel/Initializer/random_uniform/shape*
T0*
dtype0*
seed2 *

seed *0
_class&
$"loc:@discriminator/conv2d_5/kernel*(
_output_shapes
:??
?
<discriminator/conv2d_5/kernel/Initializer/random_uniform/subSub<discriminator/conv2d_5/kernel/Initializer/random_uniform/max<discriminator/conv2d_5/kernel/Initializer/random_uniform/min*
T0*0
_class&
$"loc:@discriminator/conv2d_5/kernel*
_output_shapes
: 
?
<discriminator/conv2d_5/kernel/Initializer/random_uniform/mulMulFdiscriminator/conv2d_5/kernel/Initializer/random_uniform/RandomUniform<discriminator/conv2d_5/kernel/Initializer/random_uniform/sub*
T0*0
_class&
$"loc:@discriminator/conv2d_5/kernel*(
_output_shapes
:??
?
8discriminator/conv2d_5/kernel/Initializer/random_uniformAdd<discriminator/conv2d_5/kernel/Initializer/random_uniform/mul<discriminator/conv2d_5/kernel/Initializer/random_uniform/min*
T0*0
_class&
$"loc:@discriminator/conv2d_5/kernel*(
_output_shapes
:??
?
discriminator/conv2d_5/kernel
VariableV2*
dtype0*
shared_name *
shape:??*
	container *0
_class&
$"loc:@discriminator/conv2d_5/kernel*(
_output_shapes
:??
?
$discriminator/conv2d_5/kernel/AssignAssigndiscriminator/conv2d_5/kernel8discriminator/conv2d_5/kernel/Initializer/random_uniform*
T0*
use_locking(*
validate_shape(*0
_class&
$"loc:@discriminator/conv2d_5/kernel*(
_output_shapes
:??
?
"discriminator/conv2d_5/kernel/readIdentitydiscriminator/conv2d_5/kernel*
T0*0
_class&
$"loc:@discriminator/conv2d_5/kernel*(
_output_shapes
:??
?
-discriminator/conv2d_5/bias/Initializer/zerosConst*
valueB?*    *
dtype0*.
_class$
" loc:@discriminator/conv2d_5/bias*
_output_shapes	
:?
?
discriminator/conv2d_5/bias
VariableV2*
dtype0*
shared_name *
shape:?*
	container *.
_class$
" loc:@discriminator/conv2d_5/bias*
_output_shapes	
:?
?
"discriminator/conv2d_5/bias/AssignAssigndiscriminator/conv2d_5/bias-discriminator/conv2d_5/bias/Initializer/zeros*
T0*
use_locking(*
validate_shape(*.
_class$
" loc:@discriminator/conv2d_5/bias*
_output_shapes	
:?
?
 discriminator/conv2d_5/bias/readIdentitydiscriminator/conv2d_5/bias*
T0*.
_class$
" loc:@discriminator/conv2d_5/bias*
_output_shapes	
:?
u
$discriminator/conv2d_5/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
discriminator/conv2d_5/Conv2DConv2Ddiscriminator/Relu_2"discriminator/conv2d_5/kernel/read*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingVALID*
use_cudnn_on_gpu(*0
_output_shapes
:??????????
?
discriminator/conv2d_5/BiasAddBiasAdddiscriminator/conv2d_5/Conv2D discriminator/conv2d_5/bias/read*
T0*
data_formatNHWC*0
_output_shapes
:??????????
~
discriminator/conv2d_5/ReluReludiscriminator/conv2d_5/BiasAdd*
T0*0
_output_shapes
:??????????
u
$discriminator/Mean/reduction_indicesConst*
valueB"      *
dtype0*
_output_shapes
:
?
discriminator/MeanMeandiscriminator/conv2d_5/Relu$discriminator/Mean/reduction_indices*
	keep_dims( *
T0*

Tidx0*(
_output_shapes
:??????????
?
;discriminator/dense/kernel/Initializer/random_uniform/shapeConst*
valueB"?   ?   *
dtype0*-
_class#
!loc:@discriminator/dense/kernel*
_output_shapes
:
?
9discriminator/dense/kernel/Initializer/random_uniform/minConst*
valueB
 *~5?*
dtype0*-
_class#
!loc:@discriminator/dense/kernel*
_output_shapes
: 
?
9discriminator/dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *~5>*
dtype0*-
_class#
!loc:@discriminator/dense/kernel*
_output_shapes
: 
?
Cdiscriminator/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform;discriminator/dense/kernel/Initializer/random_uniform/shape*
T0*
dtype0*
seed2 *

seed *-
_class#
!loc:@discriminator/dense/kernel*
_output_shapes
:	??
?
9discriminator/dense/kernel/Initializer/random_uniform/subSub9discriminator/dense/kernel/Initializer/random_uniform/max9discriminator/dense/kernel/Initializer/random_uniform/min*
T0*-
_class#
!loc:@discriminator/dense/kernel*
_output_shapes
: 
?
9discriminator/dense/kernel/Initializer/random_uniform/mulMulCdiscriminator/dense/kernel/Initializer/random_uniform/RandomUniform9discriminator/dense/kernel/Initializer/random_uniform/sub*
T0*-
_class#
!loc:@discriminator/dense/kernel*
_output_shapes
:	??
?
5discriminator/dense/kernel/Initializer/random_uniformAdd9discriminator/dense/kernel/Initializer/random_uniform/mul9discriminator/dense/kernel/Initializer/random_uniform/min*
T0*-
_class#
!loc:@discriminator/dense/kernel*
_output_shapes
:	??
?
discriminator/dense/kernel
VariableV2*
dtype0*
shared_name *
shape:	??*
	container *-
_class#
!loc:@discriminator/dense/kernel*
_output_shapes
:	??
?
!discriminator/dense/kernel/AssignAssigndiscriminator/dense/kernel5discriminator/dense/kernel/Initializer/random_uniform*
T0*
use_locking(*
validate_shape(*-
_class#
!loc:@discriminator/dense/kernel*
_output_shapes
:	??
?
discriminator/dense/kernel/readIdentitydiscriminator/dense/kernel*
T0*-
_class#
!loc:@discriminator/dense/kernel*
_output_shapes
:	??
?
*discriminator/dense/bias/Initializer/zerosConst*
valueB?*    *
dtype0*+
_class!
loc:@discriminator/dense/bias*
_output_shapes
:?
?
discriminator/dense/bias
VariableV2*
dtype0*
shared_name *
shape:?*
	container *+
_class!
loc:@discriminator/dense/bias*
_output_shapes
:?
?
discriminator/dense/bias/AssignAssigndiscriminator/dense/bias*discriminator/dense/bias/Initializer/zeros*
T0*
use_locking(*
validate_shape(*+
_class!
loc:@discriminator/dense/bias*
_output_shapes
:?
?
discriminator/dense/bias/readIdentitydiscriminator/dense/bias*
T0*+
_class!
loc:@discriminator/dense/bias*
_output_shapes
:?
?
discriminator/dense/MatMulMatMuldiscriminator/Meandiscriminator/dense/kernel/read*
T0*
transpose_b( *
transpose_a( *'
_output_shapes
:??????????
?
discriminator/dense/BiasAddBiasAdddiscriminator/dense/MatMuldiscriminator/dense/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:??????????
e
#discriminator/Max/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
?
discriminator/MaxMaxdiscriminator/dense/BiasAdd#discriminator/Max/reduction_indices*
	keep_dims(*
T0*

Tidx0*'
_output_shapes
:?????????
z
discriminator/subSubdiscriminator/dense/BiasAdddiscriminator/Max*
T0*'
_output_shapes
:??????????
]
discriminator/ExpExpdiscriminator/sub*
T0*'
_output_shapes
:??????????
e
#discriminator/Sum/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
?
discriminator/SumSumdiscriminator/Exp#discriminator/Sum/reduction_indices*
	keep_dims( *
T0*

Tidx0*#
_output_shapes
:?????????
Y
discriminator/LogLogdiscriminator/Sum*
T0*#
_output_shapes
:?????????
j
discriminator/SqueezeSqueezediscriminator/Max*
T0*
squeeze_dims
 *
_output_shapes
:
e
discriminator/addAdddiscriminator/Logdiscriminator/Squeeze*
T0*
_output_shapes
:
Z
discriminator/sub_1/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
g
discriminator/sub_1Subdiscriminator/adddiscriminator/sub_1/y*
T0*
_output_shapes
:
k
discriminator/outSoftmaxdiscriminator/dense/BiasAdd*
T0*'
_output_shapes
:??????????
^
discriminator_1/truediv/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
i
discriminator_1/truedivRealDiv	drop_ratediscriminator_1/truediv/y*
T0*
_output_shapes
: 
v
 discriminator_1/dropout/IdentityIdentitygenerator/Tanh*
T0*/
_output_shapes
:?????????  
u
$discriminator_1/conv2d/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
discriminator_1/conv2d/Conv2DConv2D discriminator_1/dropout/Identity discriminator/conv2d/kernel/read*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(*/
_output_shapes
:?????????@
?
discriminator_1/conv2d/BiasAddBiasAdddiscriminator_1/conv2d/Conv2Ddiscriminator/conv2d/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:?????????@
}
discriminator_1/conv2d/ReluReludiscriminator_1/conv2d/BiasAdd*
T0*/
_output_shapes
:?????????@
?
"discriminator_1/dropout_1/IdentityIdentitydiscriminator_1/conv2d/Relu*
T0*/
_output_shapes
:?????????@
w
&discriminator_1/conv2d_1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
discriminator_1/conv2d_1/Conv2DConv2D"discriminator_1/dropout_1/Identity"discriminator/conv2d_1/kernel/read*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(*/
_output_shapes
:?????????@
?
 discriminator_1/conv2d_1/BiasAddBiasAdddiscriminator_1/conv2d_1/Conv2D discriminator/conv2d_1/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:?????????@
?
discriminator_1/conv2d_1/ReluRelu discriminator_1/conv2d_1/BiasAdd*
T0*/
_output_shapes
:?????????@
w
&discriminator_1/conv2d_2/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
discriminator_1/conv2d_2/Conv2DConv2Ddiscriminator_1/conv2d_1/Relu"discriminator/conv2d_2/kernel/read*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(*/
_output_shapes
:?????????@
?
 discriminator_1/conv2d_2/BiasAddBiasAdddiscriminator_1/conv2d_2/Conv2D discriminator/conv2d_2/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:?????????@
?
2discriminator_1/batch_normalization/FusedBatchNormFusedBatchNorm discriminator_1/conv2d_2/BiasAdd,discriminator/batch_normalization/gamma/read+discriminator/batch_normalization/beta/read2discriminator/batch_normalization/moving_mean/read6discriminator/batch_normalization/moving_variance/read*
is_training( *
T0*
data_formatNHWC*
epsilon%o?:*G
_output_shapes5
3:?????????@:@:@:@:@
n
)discriminator_1/batch_normalization/ConstConst*
valueB
 *?p}?*
dtype0*
_output_shapes
: 
?
discriminator_1/ReluRelu2discriminator_1/batch_normalization/FusedBatchNorm*
T0*/
_output_shapes
:?????????@
~
"discriminator_1/dropout_2/IdentityIdentitydiscriminator_1/Relu*
T0*/
_output_shapes
:?????????@
w
&discriminator_1/conv2d_3/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
discriminator_1/conv2d_3/Conv2DConv2D"discriminator_1/dropout_2/Identity"discriminator/conv2d_3/kernel/read*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(*0
_output_shapes
:??????????
?
 discriminator_1/conv2d_3/BiasAddBiasAdddiscriminator_1/conv2d_3/Conv2D discriminator/conv2d_3/bias/read*
T0*
data_formatNHWC*0
_output_shapes
:??????????
?
4discriminator_1/batch_normalization_1/FusedBatchNormFusedBatchNorm discriminator_1/conv2d_3/BiasAdd.discriminator/batch_normalization_1/gamma/read-discriminator/batch_normalization_1/beta/read4discriminator/batch_normalization_1/moving_mean/read8discriminator/batch_normalization_1/moving_variance/read*
is_training( *
T0*
data_formatNHWC*
epsilon%o?:*L
_output_shapes:
8:??????????:?:?:?:?
p
+discriminator_1/batch_normalization_1/ConstConst*
valueB
 *?p}?*
dtype0*
_output_shapes
: 
?
discriminator_1/Relu_1Relu4discriminator_1/batch_normalization_1/FusedBatchNorm*
T0*0
_output_shapes
:??????????
w
&discriminator_1/conv2d_4/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
discriminator_1/conv2d_4/Conv2DConv2Ddiscriminator_1/Relu_1"discriminator/conv2d_4/kernel/read*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(*0
_output_shapes
:??????????
?
 discriminator_1/conv2d_4/BiasAddBiasAdddiscriminator_1/conv2d_4/Conv2D discriminator/conv2d_4/bias/read*
T0*
data_formatNHWC*0
_output_shapes
:??????????
?
4discriminator_1/batch_normalization_2/FusedBatchNormFusedBatchNorm discriminator_1/conv2d_4/BiasAdd.discriminator/batch_normalization_2/gamma/read-discriminator/batch_normalization_2/beta/read4discriminator/batch_normalization_2/moving_mean/read8discriminator/batch_normalization_2/moving_variance/read*
is_training( *
T0*
data_formatNHWC*
epsilon%o?:*L
_output_shapes:
8:??????????:?:?:?:?
p
+discriminator_1/batch_normalization_2/ConstConst*
valueB
 *?p}?*
dtype0*
_output_shapes
: 
?
discriminator_1/Relu_2Relu4discriminator_1/batch_normalization_2/FusedBatchNorm*
T0*0
_output_shapes
:??????????
w
&discriminator_1/conv2d_5/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
discriminator_1/conv2d_5/Conv2DConv2Ddiscriminator_1/Relu_2"discriminator/conv2d_5/kernel/read*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingVALID*
use_cudnn_on_gpu(*0
_output_shapes
:??????????
?
 discriminator_1/conv2d_5/BiasAddBiasAdddiscriminator_1/conv2d_5/Conv2D discriminator/conv2d_5/bias/read*
T0*
data_formatNHWC*0
_output_shapes
:??????????
?
discriminator_1/conv2d_5/ReluRelu discriminator_1/conv2d_5/BiasAdd*
T0*0
_output_shapes
:??????????
w
&discriminator_1/Mean/reduction_indicesConst*
valueB"      *
dtype0*
_output_shapes
:
?
discriminator_1/MeanMeandiscriminator_1/conv2d_5/Relu&discriminator_1/Mean/reduction_indices*
	keep_dims( *
T0*

Tidx0*(
_output_shapes
:??????????
?
discriminator_1/dense/MatMulMatMuldiscriminator_1/Meandiscriminator/dense/kernel/read*
T0*
transpose_b( *
transpose_a( *'
_output_shapes
:??????????
?
discriminator_1/dense/BiasAddBiasAdddiscriminator_1/dense/MatMuldiscriminator/dense/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:??????????
g
%discriminator_1/Max/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
?
discriminator_1/MaxMaxdiscriminator_1/dense/BiasAdd%discriminator_1/Max/reduction_indices*
	keep_dims(*
T0*

Tidx0*'
_output_shapes
:?????????
?
discriminator_1/subSubdiscriminator_1/dense/BiasAdddiscriminator_1/Max*
T0*'
_output_shapes
:??????????
a
discriminator_1/ExpExpdiscriminator_1/sub*
T0*'
_output_shapes
:??????????
g
%discriminator_1/Sum/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
?
discriminator_1/SumSumdiscriminator_1/Exp%discriminator_1/Sum/reduction_indices*
	keep_dims( *
T0*

Tidx0*#
_output_shapes
:?????????
]
discriminator_1/LogLogdiscriminator_1/Sum*
T0*#
_output_shapes
:?????????
n
discriminator_1/SqueezeSqueezediscriminator_1/Max*
T0*
squeeze_dims
 *
_output_shapes
:
k
discriminator_1/addAdddiscriminator_1/Logdiscriminator_1/Squeeze*
T0*
_output_shapes
:
\
discriminator_1/sub_1/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
m
discriminator_1/sub_1Subdiscriminator_1/adddiscriminator_1/sub_1/y*
T0*
_output_shapes
:
o
discriminator_1/outSoftmaxdiscriminator_1/dense/BiasAdd*
T0*'
_output_shapes
:??????????
k
ones_like/ShapeShapediscriminator/sub_1*
T0*
out_type0*#
_output_shapes
:?????????
T
ones_like/ConstConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
h
	ones_likeFillones_like/Shapeones_like/Const*
T0*

index_type0*
_output_shapes
:
]
logistic_loss/zeros_like	ZerosLikediscriminator/sub_1*
T0*
_output_shapes
:
|
logistic_loss/GreaterEqualGreaterEqualdiscriminator/sub_1logistic_loss/zeros_like*
T0*
_output_shapes
:
?
logistic_loss/SelectSelectlogistic_loss/GreaterEqualdiscriminator/sub_1logistic_loss/zeros_like*
T0*
_output_shapes
:
P
logistic_loss/NegNegdiscriminator/sub_1*
T0*
_output_shapes
:
?
logistic_loss/Select_1Selectlogistic_loss/GreaterEquallogistic_loss/Negdiscriminator/sub_1*
T0*
_output_shapes
:
[
logistic_loss/mulMuldiscriminator/sub_1	ones_like*
T0*
_output_shapes
:
d
logistic_loss/subSublogistic_loss/Selectlogistic_loss/mul*
T0*
_output_shapes
:
S
logistic_loss/ExpExplogistic_loss/Select_1*
T0*
_output_shapes
:
R
logistic_loss/Log1pLog1plogistic_loss/Exp*
T0*
_output_shapes
:
_
logistic_lossAddlogistic_loss/sublogistic_loss/Log1p*
T0*
_output_shapes
:
<
RankRanklogistic_loss*
T0*
_output_shapes
: 
M
range/startConst*
value	B : *
dtype0*
_output_shapes
: 
M
range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
_
rangeRangerange/startRankrange/delta*

Tidx0*#
_output_shapes
:?????????
`
MeanMeanlogistic_lossrange*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
Q

zeros_like	ZerosLikediscriminator_1/sub_1*
T0*
_output_shapes
:
a
logistic_loss_1/zeros_like	ZerosLikediscriminator_1/sub_1*
T0*
_output_shapes
:
?
logistic_loss_1/GreaterEqualGreaterEqualdiscriminator_1/sub_1logistic_loss_1/zeros_like*
T0*
_output_shapes
:
?
logistic_loss_1/SelectSelectlogistic_loss_1/GreaterEqualdiscriminator_1/sub_1logistic_loss_1/zeros_like*
T0*
_output_shapes
:
T
logistic_loss_1/NegNegdiscriminator_1/sub_1*
T0*
_output_shapes
:
?
logistic_loss_1/Select_1Selectlogistic_loss_1/GreaterEquallogistic_loss_1/Negdiscriminator_1/sub_1*
T0*
_output_shapes
:
`
logistic_loss_1/mulMuldiscriminator_1/sub_1
zeros_like*
T0*
_output_shapes
:
j
logistic_loss_1/subSublogistic_loss_1/Selectlogistic_loss_1/mul*
T0*
_output_shapes
:
W
logistic_loss_1/ExpExplogistic_loss_1/Select_1*
T0*
_output_shapes
:
V
logistic_loss_1/Log1pLog1plogistic_loss_1/Exp*
T0*
_output_shapes
:
e
logistic_loss_1Addlogistic_loss_1/sublogistic_loss_1/Log1p*
T0*
_output_shapes
:
@
Rank_1Ranklogistic_loss_1*
T0*
_output_shapes
: 
O
range_1/startConst*
value	B : *
dtype0*
_output_shapes
: 
O
range_1/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
g
range_1Rangerange_1/startRank_1range_1/delta*

Tidx0*#
_output_shapes
:?????????
f
Mean_1Meanlogistic_loss_1range_1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
L
SqueezeSqueezey*
T0*
squeeze_dims
 *
_output_shapes
:
U
one_hot/on_valueConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
V
one_hot/off_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
O
one_hot/depthConst*
value	B :?*
dtype0*
_output_shapes
: 
?
one_hotOneHotSqueezeone_hot/depthone_hot/on_valueone_hot/off_value*
axis?????????*
T0*
TI0*
_output_shapes
:
u
9softmax_cross_entropy_with_logits_sg/labels_stop_gradientStopGradientone_hot*
T0*
_output_shapes
:
k
)softmax_cross_entropy_with_logits_sg/RankConst*
value	B :*
dtype0*
_output_shapes
: 
?
*softmax_cross_entropy_with_logits_sg/ShapeShapediscriminator/dense/BiasAdd*
T0*
out_type0*
_output_shapes
:
m
+softmax_cross_entropy_with_logits_sg/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
?
,softmax_cross_entropy_with_logits_sg/Shape_1Shapediscriminator/dense/BiasAdd*
T0*
out_type0*
_output_shapes
:
l
*softmax_cross_entropy_with_logits_sg/Sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
?
(softmax_cross_entropy_with_logits_sg/SubSub+softmax_cross_entropy_with_logits_sg/Rank_1*softmax_cross_entropy_with_logits_sg/Sub/y*
T0*
_output_shapes
: 
?
0softmax_cross_entropy_with_logits_sg/Slice/beginPack(softmax_cross_entropy_with_logits_sg/Sub*

axis *
T0*
N*
_output_shapes
:
y
/softmax_cross_entropy_with_logits_sg/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
?
*softmax_cross_entropy_with_logits_sg/SliceSlice,softmax_cross_entropy_with_logits_sg/Shape_10softmax_cross_entropy_with_logits_sg/Slice/begin/softmax_cross_entropy_with_logits_sg/Slice/size*
Index0*
T0*
_output_shapes
:
?
4softmax_cross_entropy_with_logits_sg/concat/values_0Const*
valueB:
?????????*
dtype0*
_output_shapes
:
r
0softmax_cross_entropy_with_logits_sg/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
+softmax_cross_entropy_with_logits_sg/concatConcatV24softmax_cross_entropy_with_logits_sg/concat/values_0*softmax_cross_entropy_with_logits_sg/Slice0softmax_cross_entropy_with_logits_sg/concat/axis*
T0*
N*

Tidx0*
_output_shapes
:
?
,softmax_cross_entropy_with_logits_sg/ReshapeReshapediscriminator/dense/BiasAdd+softmax_cross_entropy_with_logits_sg/concat*
T0*
Tshape0*0
_output_shapes
:??????????????????
?
+softmax_cross_entropy_with_logits_sg/Rank_2Rank9softmax_cross_entropy_with_logits_sg/labels_stop_gradient*
T0*
_output_shapes
: 
?
,softmax_cross_entropy_with_logits_sg/Shape_2Shape9softmax_cross_entropy_with_logits_sg/labels_stop_gradient*
T0*
out_type0*#
_output_shapes
:?????????
n
,softmax_cross_entropy_with_logits_sg/Sub_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
?
*softmax_cross_entropy_with_logits_sg/Sub_1Sub+softmax_cross_entropy_with_logits_sg/Rank_2,softmax_cross_entropy_with_logits_sg/Sub_1/y*
T0*
_output_shapes
: 
?
2softmax_cross_entropy_with_logits_sg/Slice_1/beginPack*softmax_cross_entropy_with_logits_sg/Sub_1*

axis *
T0*
N*
_output_shapes
:
{
1softmax_cross_entropy_with_logits_sg/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
?
,softmax_cross_entropy_with_logits_sg/Slice_1Slice,softmax_cross_entropy_with_logits_sg/Shape_22softmax_cross_entropy_with_logits_sg/Slice_1/begin1softmax_cross_entropy_with_logits_sg/Slice_1/size*
Index0*
T0*
_output_shapes
:
?
6softmax_cross_entropy_with_logits_sg/concat_1/values_0Const*
valueB:
?????????*
dtype0*
_output_shapes
:
t
2softmax_cross_entropy_with_logits_sg/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
-softmax_cross_entropy_with_logits_sg/concat_1ConcatV26softmax_cross_entropy_with_logits_sg/concat_1/values_0,softmax_cross_entropy_with_logits_sg/Slice_12softmax_cross_entropy_with_logits_sg/concat_1/axis*
T0*
N*

Tidx0*
_output_shapes
:
?
.softmax_cross_entropy_with_logits_sg/Reshape_1Reshape9softmax_cross_entropy_with_logits_sg/labels_stop_gradient-softmax_cross_entropy_with_logits_sg/concat_1*
T0*
Tshape0*0
_output_shapes
:??????????????????
?
$softmax_cross_entropy_with_logits_sgSoftmaxCrossEntropyWithLogits,softmax_cross_entropy_with_logits_sg/Reshape.softmax_cross_entropy_with_logits_sg/Reshape_1*
T0*?
_output_shapes-
+:?????????:??????????????????
n
,softmax_cross_entropy_with_logits_sg/Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
?
*softmax_cross_entropy_with_logits_sg/Sub_2Sub)softmax_cross_entropy_with_logits_sg/Rank,softmax_cross_entropy_with_logits_sg/Sub_2/y*
T0*
_output_shapes
: 
|
2softmax_cross_entropy_with_logits_sg/Slice_2/beginConst*
valueB: *
dtype0*
_output_shapes
:
?
1softmax_cross_entropy_with_logits_sg/Slice_2/sizePack*softmax_cross_entropy_with_logits_sg/Sub_2*

axis *
T0*
N*
_output_shapes
:
?
,softmax_cross_entropy_with_logits_sg/Slice_2Slice*softmax_cross_entropy_with_logits_sg/Shape2softmax_cross_entropy_with_logits_sg/Slice_2/begin1softmax_cross_entropy_with_logits_sg/Slice_2/size*
Index0*
T0*
_output_shapes
:
?
.softmax_cross_entropy_with_logits_sg/Reshape_2Reshape$softmax_cross_entropy_with_logits_sg,softmax_cross_entropy_with_logits_sg/Slice_2*
T0*
Tshape0*#
_output_shapes
:?????????
{
	Squeeze_1Squeeze.softmax_cross_entropy_with_logits_sg/Reshape_2*
T0*
squeeze_dims
 *
_output_shapes
:
]
ToFloatCast
label_mask*

DstT0*
Truncate( *

SrcT0*
_output_shapes
:
T
	Squeeze_2SqueezeToFloat*
T0*
squeeze_dims
 *
_output_shapes
:
C
mulMul	Squeeze_2	Squeeze_1*
T0*
_output_shapes
:
4
Rank_2Rankmul*
T0*
_output_shapes
: 
O
range_2/startConst*
value	B : *
dtype0*
_output_shapes
: 
O
range_2/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
g
range_2Rangerange_2/startRank_2range_2/delta*

Tidx0*#
_output_shapes
:?????????
V
SumSummulrange_2*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
:
Rank_3Rank	Squeeze_2*
T0*
_output_shapes
: 
O
range_3/startConst*
value	B : *
dtype0*
_output_shapes
: 
O
range_3/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
g
range_3Rangerange_3/startRank_3range_3/delta*

Tidx0*#
_output_shapes
:?????????
^
Sum_1Sum	Squeeze_2range_3*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
N
	Maximum/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
E
MaximumMaximum	Maximum/xSum_1*
T0*
_output_shapes
: 
A
truedivRealDivSumMaximum*
T0*
_output_shapes
: 
:
addAddtruedivMean*
T0*
_output_shapes
: 
:
add_1AddaddMean_1*
T0*
_output_shapes
: 
Z
Mean_2/reduction_indicesConst*
value	B : *
dtype0*
_output_shapes
: 

Mean_2Meandiscriminator/MeanMean_2/reduction_indices*
	keep_dims( *
T0*

Tidx0*
_output_shapes	
:?
Z
Mean_3/reduction_indicesConst*
value	B : *
dtype0*
_output_shapes
: 
?
Mean_3Meandiscriminator_1/MeanMean_3/reduction_indices*
	keep_dims( *
T0*

Tidx0*
_output_shapes	
:?
@
subSubMean_2Mean_3*
T0*
_output_shapes	
:?
5
AbsAbssub*
T0*
_output_shapes	
:?
O
ConstConst*
valueB: *
dtype0*
_output_shapes
:
X
Mean_4MeanAbsConst*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
R
ArgMax/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
?
ArgMaxArgMaxdiscriminator/dense/BiasAddArgMax/dimension*
output_type0	*
T0*

Tidx0*#
_output_shapes
:?????????
a
CastCastArgMax*

DstT0*
Truncate( *

SrcT0	*#
_output_shapes
:?????????
T
	Squeeze_3SqueezeSqueeze*
T0*
squeeze_dims
 *
_output_shapes
:
B
EqualEqual	Squeeze_3Cast*
T0*
_output_shapes
:
Z
	ToFloat_1CastEqual*

DstT0*
Truncate( *

SrcT0
*
_output_shapes
:
:
Rank_4Rank	ToFloat_1*
T0*
_output_shapes
: 
O
range_4/startConst*
value	B : *
dtype0*
_output_shapes
: 
O
range_4/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
g
range_4Rangerange_4/startRank_4range_4/delta*

Tidx0*#
_output_shapes
:?????????
^
Sum_2Sum	ToFloat_1range_4*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
Z
	ToFloat_2CastEqual*

DstT0*
Truncate( *

SrcT0
*
_output_shapes
:
E
mul_1Mul	Squeeze_2	ToFloat_2*
T0*
_output_shapes
:
6
Rank_5Rankmul_1*
T0*
_output_shapes
: 
O
range_5/startConst*
value	B : *
dtype0*
_output_shapes
: 
O
range_5/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
g
range_5Rangerange_5/startRank_5range_5/delta*

Tidx0*#
_output_shapes
:?????????
Z
Sum_3Summul_1range_5*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
X
gradients/grad_ys_0Const*
valueB
 *  ??*
dtype0*
_output_shapes
: 
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
>
%gradients/add_1_grad/tuple/group_depsNoOp^gradients/Fill
?
-gradients/add_1_grad/tuple/control_dependencyIdentitygradients/Fill&^gradients/add_1_grad/tuple/group_deps*
T0*!
_class
loc:@gradients/Fill*
_output_shapes
: 
?
/gradients/add_1_grad/tuple/control_dependency_1Identitygradients/Fill&^gradients/add_1_grad/tuple/group_deps*
T0*!
_class
loc:@gradients/Fill*
_output_shapes
: 
[
#gradients/add_grad/tuple/group_depsNoOp.^gradients/add_1_grad/tuple/control_dependency
?
+gradients/add_grad/tuple/control_dependencyIdentity-gradients/add_1_grad/tuple/control_dependency$^gradients/add_grad/tuple/group_deps*
T0*!
_class
loc:@gradients/Fill*
_output_shapes
: 
?
-gradients/add_grad/tuple/control_dependency_1Identity-gradients/add_1_grad/tuple/control_dependency$^gradients/add_grad/tuple/group_deps*
T0*!
_class
loc:@gradients/Fill*
_output_shapes
: 
s
gradients/Mean_1_grad/ShapeShapelogistic_loss_1*
T0*
out_type0*#
_output_shapes
:?????????
?
gradients/Mean_1_grad/SizeSizegradients/Mean_1_grad/Shape*
T0*
out_type0*.
_class$
" loc:@gradients/Mean_1_grad/Shape*
_output_shapes
: 
?
gradients/Mean_1_grad/addAddrange_1gradients/Mean_1_grad/Size*
T0*.
_class$
" loc:@gradients/Mean_1_grad/Shape*#
_output_shapes
:?????????
?
gradients/Mean_1_grad/modFloorModgradients/Mean_1_grad/addgradients/Mean_1_grad/Size*
T0*.
_class$
" loc:@gradients/Mean_1_grad/Shape*#
_output_shapes
:?????????
?
gradients/Mean_1_grad/Shape_1Shapegradients/Mean_1_grad/mod*
T0*
out_type0*.
_class$
" loc:@gradients/Mean_1_grad/Shape*
_output_shapes
:
?
!gradients/Mean_1_grad/range/startConst*
value	B : *
dtype0*.
_class$
" loc:@gradients/Mean_1_grad/Shape*
_output_shapes
: 
?
!gradients/Mean_1_grad/range/deltaConst*
value	B :*
dtype0*.
_class$
" loc:@gradients/Mean_1_grad/Shape*
_output_shapes
: 
?
gradients/Mean_1_grad/rangeRange!gradients/Mean_1_grad/range/startgradients/Mean_1_grad/Size!gradients/Mean_1_grad/range/delta*

Tidx0*.
_class$
" loc:@gradients/Mean_1_grad/Shape*#
_output_shapes
:?????????
?
 gradients/Mean_1_grad/Fill/valueConst*
value	B :*
dtype0*.
_class$
" loc:@gradients/Mean_1_grad/Shape*
_output_shapes
: 
?
gradients/Mean_1_grad/FillFillgradients/Mean_1_grad/Shape_1 gradients/Mean_1_grad/Fill/value*
T0*

index_type0*.
_class$
" loc:@gradients/Mean_1_grad/Shape*#
_output_shapes
:?????????
?
#gradients/Mean_1_grad/DynamicStitchDynamicStitchgradients/Mean_1_grad/rangegradients/Mean_1_grad/modgradients/Mean_1_grad/Shapegradients/Mean_1_grad/Fill*
T0*
N*.
_class$
" loc:@gradients/Mean_1_grad/Shape*#
_output_shapes
:?????????
?
gradients/Mean_1_grad/Maximum/yConst*
value	B :*
dtype0*.
_class$
" loc:@gradients/Mean_1_grad/Shape*
_output_shapes
: 
?
gradients/Mean_1_grad/MaximumMaximum#gradients/Mean_1_grad/DynamicStitchgradients/Mean_1_grad/Maximum/y*
T0*.
_class$
" loc:@gradients/Mean_1_grad/Shape*#
_output_shapes
:?????????
?
gradients/Mean_1_grad/floordivFloorDivgradients/Mean_1_grad/Shapegradients/Mean_1_grad/Maximum*
T0*.
_class$
" loc:@gradients/Mean_1_grad/Shape*#
_output_shapes
:?????????
?
gradients/Mean_1_grad/ReshapeReshape/gradients/add_1_grad/tuple/control_dependency_1#gradients/Mean_1_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
?
gradients/Mean_1_grad/TileTilegradients/Mean_1_grad/Reshapegradients/Mean_1_grad/floordiv*
T0*

Tmultiples0*
_output_shapes
:
u
gradients/Mean_1_grad/Shape_2Shapelogistic_loss_1*
T0*
out_type0*#
_output_shapes
:?????????
`
gradients/Mean_1_grad/Shape_3Const*
valueB *
dtype0*
_output_shapes
: 
e
gradients/Mean_1_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
?
gradients/Mean_1_grad/ProdProdgradients/Mean_1_grad/Shape_2gradients/Mean_1_grad/Const*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
g
gradients/Mean_1_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
?
gradients/Mean_1_grad/Prod_1Prodgradients/Mean_1_grad/Shape_3gradients/Mean_1_grad/Const_1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
c
!gradients/Mean_1_grad/Maximum_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
?
gradients/Mean_1_grad/Maximum_1Maximumgradients/Mean_1_grad/Prod_1!gradients/Mean_1_grad/Maximum_1/y*
T0*
_output_shapes
: 
?
 gradients/Mean_1_grad/floordiv_1FloorDivgradients/Mean_1_grad/Prodgradients/Mean_1_grad/Maximum_1*
T0*
_output_shapes
: 
?
gradients/Mean_1_grad/CastCast gradients/Mean_1_grad/floordiv_1*

DstT0*
Truncate( *

SrcT0*
_output_shapes
: 
?
gradients/Mean_1_grad/truedivRealDivgradients/Mean_1_grad/Tilegradients/Mean_1_grad/Cast*
T0*
_output_shapes
:
_
gradients/truediv_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
a
gradients/truediv_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
?
,gradients/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/truediv_grad/Shapegradients/truediv_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
gradients/truediv_grad/RealDivRealDiv+gradients/add_grad/tuple/control_dependencyMaximum*
T0*
_output_shapes
: 
?
gradients/truediv_grad/SumSumgradients/truediv_grad/RealDiv,gradients/truediv_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
?
gradients/truediv_grad/ReshapeReshapegradients/truediv_grad/Sumgradients/truediv_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
G
gradients/truediv_grad/NegNegSum*
T0*
_output_shapes
: 
q
 gradients/truediv_grad/RealDiv_1RealDivgradients/truediv_grad/NegMaximum*
T0*
_output_shapes
: 
w
 gradients/truediv_grad/RealDiv_2RealDiv gradients/truediv_grad/RealDiv_1Maximum*
T0*
_output_shapes
: 
?
gradients/truediv_grad/mulMul+gradients/add_grad/tuple/control_dependency gradients/truediv_grad/RealDiv_2*
T0*
_output_shapes
: 
?
gradients/truediv_grad/Sum_1Sumgradients/truediv_grad/mul.gradients/truediv_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
?
 gradients/truediv_grad/Reshape_1Reshapegradients/truediv_grad/Sum_1gradients/truediv_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
s
'gradients/truediv_grad/tuple/group_depsNoOp^gradients/truediv_grad/Reshape!^gradients/truediv_grad/Reshape_1
?
/gradients/truediv_grad/tuple/control_dependencyIdentitygradients/truediv_grad/Reshape(^gradients/truediv_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/truediv_grad/Reshape*
_output_shapes
: 
?
1gradients/truediv_grad/tuple/control_dependency_1Identity gradients/truediv_grad/Reshape_1(^gradients/truediv_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/truediv_grad/Reshape_1*
_output_shapes
: 
o
gradients/Mean_grad/ShapeShapelogistic_loss*
T0*
out_type0*#
_output_shapes
:?????????
?
gradients/Mean_grad/SizeSizegradients/Mean_grad/Shape*
T0*
out_type0*,
_class"
 loc:@gradients/Mean_grad/Shape*
_output_shapes
: 
?
gradients/Mean_grad/addAddrangegradients/Mean_grad/Size*
T0*,
_class"
 loc:@gradients/Mean_grad/Shape*#
_output_shapes
:?????????
?
gradients/Mean_grad/modFloorModgradients/Mean_grad/addgradients/Mean_grad/Size*
T0*,
_class"
 loc:@gradients/Mean_grad/Shape*#
_output_shapes
:?????????
?
gradients/Mean_grad/Shape_1Shapegradients/Mean_grad/mod*
T0*
out_type0*,
_class"
 loc:@gradients/Mean_grad/Shape*
_output_shapes
:
?
gradients/Mean_grad/range/startConst*
value	B : *
dtype0*,
_class"
 loc:@gradients/Mean_grad/Shape*
_output_shapes
: 
?
gradients/Mean_grad/range/deltaConst*
value	B :*
dtype0*,
_class"
 loc:@gradients/Mean_grad/Shape*
_output_shapes
: 
?
gradients/Mean_grad/rangeRangegradients/Mean_grad/range/startgradients/Mean_grad/Sizegradients/Mean_grad/range/delta*

Tidx0*,
_class"
 loc:@gradients/Mean_grad/Shape*#
_output_shapes
:?????????
?
gradients/Mean_grad/Fill/valueConst*
value	B :*
dtype0*,
_class"
 loc:@gradients/Mean_grad/Shape*
_output_shapes
: 
?
gradients/Mean_grad/FillFillgradients/Mean_grad/Shape_1gradients/Mean_grad/Fill/value*
T0*

index_type0*,
_class"
 loc:@gradients/Mean_grad/Shape*#
_output_shapes
:?????????
?
!gradients/Mean_grad/DynamicStitchDynamicStitchgradients/Mean_grad/rangegradients/Mean_grad/modgradients/Mean_grad/Shapegradients/Mean_grad/Fill*
T0*
N*,
_class"
 loc:@gradients/Mean_grad/Shape*#
_output_shapes
:?????????
?
gradients/Mean_grad/Maximum/yConst*
value	B :*
dtype0*,
_class"
 loc:@gradients/Mean_grad/Shape*
_output_shapes
: 
?
gradients/Mean_grad/MaximumMaximum!gradients/Mean_grad/DynamicStitchgradients/Mean_grad/Maximum/y*
T0*,
_class"
 loc:@gradients/Mean_grad/Shape*#
_output_shapes
:?????????
?
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Shapegradients/Mean_grad/Maximum*
T0*,
_class"
 loc:@gradients/Mean_grad/Shape*#
_output_shapes
:?????????
?
gradients/Mean_grad/ReshapeReshape-gradients/add_grad/tuple/control_dependency_1!gradients/Mean_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
?
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/floordiv*
T0*

Tmultiples0*
_output_shapes
:
q
gradients/Mean_grad/Shape_2Shapelogistic_loss*
T0*
out_type0*#
_output_shapes
:?????????
^
gradients/Mean_grad/Shape_3Const*
valueB *
dtype0*
_output_shapes
: 
c
gradients/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
?
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_2gradients/Mean_grad/Const*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
e
gradients/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
?
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_3gradients/Mean_grad/Const_1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
a
gradients/Mean_grad/Maximum_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
?
gradients/Mean_grad/Maximum_1Maximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum_1/y*
T0*
_output_shapes
: 
?
gradients/Mean_grad/floordiv_1FloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum_1*
T0*
_output_shapes
: 
?
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv_1*

DstT0*
Truncate( *

SrcT0*
_output_shapes
: 
}
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*
_output_shapes
:
?
$gradients/logistic_loss_1_grad/ShapeShapelogistic_loss_1/sub*
T0*
out_type0*#
_output_shapes
:?????????
?
&gradients/logistic_loss_1_grad/Shape_1Shapelogistic_loss_1/Log1p*
T0*
out_type0*#
_output_shapes
:?????????
?
4gradients/logistic_loss_1_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients/logistic_loss_1_grad/Shape&gradients/logistic_loss_1_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
"gradients/logistic_loss_1_grad/SumSumgradients/Mean_1_grad/truediv4gradients/logistic_loss_1_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
?
&gradients/logistic_loss_1_grad/ReshapeReshape"gradients/logistic_loss_1_grad/Sum$gradients/logistic_loss_1_grad/Shape*
T0*
Tshape0*
_output_shapes
:
?
$gradients/logistic_loss_1_grad/Sum_1Sumgradients/Mean_1_grad/truediv6gradients/logistic_loss_1_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
?
(gradients/logistic_loss_1_grad/Reshape_1Reshape$gradients/logistic_loss_1_grad/Sum_1&gradients/logistic_loss_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
?
/gradients/logistic_loss_1_grad/tuple/group_depsNoOp'^gradients/logistic_loss_1_grad/Reshape)^gradients/logistic_loss_1_grad/Reshape_1
?
7gradients/logistic_loss_1_grad/tuple/control_dependencyIdentity&gradients/logistic_loss_1_grad/Reshape0^gradients/logistic_loss_1_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/logistic_loss_1_grad/Reshape*
_output_shapes
:
?
9gradients/logistic_loss_1_grad/tuple/control_dependency_1Identity(gradients/logistic_loss_1_grad/Reshape_10^gradients/logistic_loss_1_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/logistic_loss_1_grad/Reshape_1*
_output_shapes
:
d
gradients/Sum_grad/ShapeShapemul*
T0*
out_type0*#
_output_shapes
:?????????
?
gradients/Sum_grad/SizeSizegradients/Sum_grad/Shape*
T0*
out_type0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
: 
?
gradients/Sum_grad/addAddrange_2gradients/Sum_grad/Size*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*#
_output_shapes
:?????????
?
gradients/Sum_grad/modFloorModgradients/Sum_grad/addgradients/Sum_grad/Size*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*#
_output_shapes
:?????????
?
gradients/Sum_grad/Shape_1Shapegradients/Sum_grad/mod*
T0*
out_type0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
:
?
gradients/Sum_grad/range/startConst*
value	B : *
dtype0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
: 
?
gradients/Sum_grad/range/deltaConst*
value	B :*
dtype0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
: 
?
gradients/Sum_grad/rangeRangegradients/Sum_grad/range/startgradients/Sum_grad/Sizegradients/Sum_grad/range/delta*

Tidx0*+
_class!
loc:@gradients/Sum_grad/Shape*#
_output_shapes
:?????????
?
gradients/Sum_grad/Fill/valueConst*
value	B :*
dtype0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
: 
?
gradients/Sum_grad/FillFillgradients/Sum_grad/Shape_1gradients/Sum_grad/Fill/value*
T0*

index_type0*+
_class!
loc:@gradients/Sum_grad/Shape*#
_output_shapes
:?????????
?
 gradients/Sum_grad/DynamicStitchDynamicStitchgradients/Sum_grad/rangegradients/Sum_grad/modgradients/Sum_grad/Shapegradients/Sum_grad/Fill*
T0*
N*+
_class!
loc:@gradients/Sum_grad/Shape*#
_output_shapes
:?????????
?
gradients/Sum_grad/Maximum/yConst*
value	B :*
dtype0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
: 
?
gradients/Sum_grad/MaximumMaximum gradients/Sum_grad/DynamicStitchgradients/Sum_grad/Maximum/y*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*#
_output_shapes
:?????????
?
gradients/Sum_grad/floordivFloorDivgradients/Sum_grad/Shapegradients/Sum_grad/Maximum*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*#
_output_shapes
:?????????
?
gradients/Sum_grad/ReshapeReshape/gradients/truediv_grad/tuple/control_dependency gradients/Sum_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
?
gradients/Sum_grad/TileTilegradients/Sum_grad/Reshapegradients/Sum_grad/floordiv*
T0*

Tmultiples0*
_output_shapes
:
|
"gradients/logistic_loss_grad/ShapeShapelogistic_loss/sub*
T0*
out_type0*#
_output_shapes
:?????????
?
$gradients/logistic_loss_grad/Shape_1Shapelogistic_loss/Log1p*
T0*
out_type0*#
_output_shapes
:?????????
?
2gradients/logistic_loss_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/logistic_loss_grad/Shape$gradients/logistic_loss_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
 gradients/logistic_loss_grad/SumSumgradients/Mean_grad/truediv2gradients/logistic_loss_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
?
$gradients/logistic_loss_grad/ReshapeReshape gradients/logistic_loss_grad/Sum"gradients/logistic_loss_grad/Shape*
T0*
Tshape0*
_output_shapes
:
?
"gradients/logistic_loss_grad/Sum_1Sumgradients/Mean_grad/truediv4gradients/logistic_loss_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
?
&gradients/logistic_loss_grad/Reshape_1Reshape"gradients/logistic_loss_grad/Sum_1$gradients/logistic_loss_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
?
-gradients/logistic_loss_grad/tuple/group_depsNoOp%^gradients/logistic_loss_grad/Reshape'^gradients/logistic_loss_grad/Reshape_1
?
5gradients/logistic_loss_grad/tuple/control_dependencyIdentity$gradients/logistic_loss_grad/Reshape.^gradients/logistic_loss_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/logistic_loss_grad/Reshape*
_output_shapes
:
?
7gradients/logistic_loss_grad/tuple/control_dependency_1Identity&gradients/logistic_loss_grad/Reshape_1.^gradients/logistic_loss_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/logistic_loss_grad/Reshape_1*
_output_shapes
:
?
(gradients/logistic_loss_1/sub_grad/ShapeShapelogistic_loss_1/Select*
T0*
out_type0*#
_output_shapes
:?????????
?
*gradients/logistic_loss_1/sub_grad/Shape_1Shapelogistic_loss_1/mul*
T0*
out_type0*#
_output_shapes
:?????????
?
8gradients/logistic_loss_1/sub_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients/logistic_loss_1/sub_grad/Shape*gradients/logistic_loss_1/sub_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
&gradients/logistic_loss_1/sub_grad/SumSum7gradients/logistic_loss_1_grad/tuple/control_dependency8gradients/logistic_loss_1/sub_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
?
*gradients/logistic_loss_1/sub_grad/ReshapeReshape&gradients/logistic_loss_1/sub_grad/Sum(gradients/logistic_loss_1/sub_grad/Shape*
T0*
Tshape0*
_output_shapes
:
?
(gradients/logistic_loss_1/sub_grad/Sum_1Sum7gradients/logistic_loss_1_grad/tuple/control_dependency:gradients/logistic_loss_1/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
z
&gradients/logistic_loss_1/sub_grad/NegNeg(gradients/logistic_loss_1/sub_grad/Sum_1*
T0*
_output_shapes
:
?
,gradients/logistic_loss_1/sub_grad/Reshape_1Reshape&gradients/logistic_loss_1/sub_grad/Neg*gradients/logistic_loss_1/sub_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
?
3gradients/logistic_loss_1/sub_grad/tuple/group_depsNoOp+^gradients/logistic_loss_1/sub_grad/Reshape-^gradients/logistic_loss_1/sub_grad/Reshape_1
?
;gradients/logistic_loss_1/sub_grad/tuple/control_dependencyIdentity*gradients/logistic_loss_1/sub_grad/Reshape4^gradients/logistic_loss_1/sub_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/logistic_loss_1/sub_grad/Reshape*
_output_shapes
:
?
=gradients/logistic_loss_1/sub_grad/tuple/control_dependency_1Identity,gradients/logistic_loss_1/sub_grad/Reshape_14^gradients/logistic_loss_1/sub_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/logistic_loss_1/sub_grad/Reshape_1*
_output_shapes
:
?
*gradients/logistic_loss_1/Log1p_grad/add/xConst:^gradients/logistic_loss_1_grad/tuple/control_dependency_1*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
(gradients/logistic_loss_1/Log1p_grad/addAdd*gradients/logistic_loss_1/Log1p_grad/add/xlogistic_loss_1/Exp*
T0*
_output_shapes
:
?
/gradients/logistic_loss_1/Log1p_grad/Reciprocal
Reciprocal(gradients/logistic_loss_1/Log1p_grad/add*
T0*
_output_shapes
:
?
(gradients/logistic_loss_1/Log1p_grad/mulMul9gradients/logistic_loss_1_grad/tuple/control_dependency_1/gradients/logistic_loss_1/Log1p_grad/Reciprocal*
T0*
_output_shapes
:
j
gradients/mul_grad/ShapeShape	Squeeze_2*
T0*
out_type0*#
_output_shapes
:?????????
l
gradients/mul_grad/Shape_1Shape	Squeeze_1*
T0*
out_type0*#
_output_shapes
:?????????
?
(gradients/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_grad/Shapegradients/mul_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
d
gradients/mul_grad/MulMulgradients/Sum_grad/Tile	Squeeze_1*
T0*
_output_shapes
:
?
gradients/mul_grad/SumSumgradients/mul_grad/Mul(gradients/mul_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
?
gradients/mul_grad/ReshapeReshapegradients/mul_grad/Sumgradients/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
:
f
gradients/mul_grad/Mul_1Mul	Squeeze_2gradients/Sum_grad/Tile*
T0*
_output_shapes
:
?
gradients/mul_grad/Sum_1Sumgradients/mul_grad/Mul_1*gradients/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
?
gradients/mul_grad/Reshape_1Reshapegradients/mul_grad/Sum_1gradients/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
g
#gradients/mul_grad/tuple/group_depsNoOp^gradients/mul_grad/Reshape^gradients/mul_grad/Reshape_1
?
+gradients/mul_grad/tuple/control_dependencyIdentitygradients/mul_grad/Reshape$^gradients/mul_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/mul_grad/Reshape*
_output_shapes
:
?
-gradients/mul_grad/tuple/control_dependency_1Identitygradients/mul_grad/Reshape_1$^gradients/mul_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/mul_grad/Reshape_1*
_output_shapes
:
?
&gradients/logistic_loss/sub_grad/ShapeShapelogistic_loss/Select*
T0*
out_type0*#
_output_shapes
:?????????
?
(gradients/logistic_loss/sub_grad/Shape_1Shapelogistic_loss/mul*
T0*
out_type0*#
_output_shapes
:?????????
?
6gradients/logistic_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients/logistic_loss/sub_grad/Shape(gradients/logistic_loss/sub_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
$gradients/logistic_loss/sub_grad/SumSum5gradients/logistic_loss_grad/tuple/control_dependency6gradients/logistic_loss/sub_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
?
(gradients/logistic_loss/sub_grad/ReshapeReshape$gradients/logistic_loss/sub_grad/Sum&gradients/logistic_loss/sub_grad/Shape*
T0*
Tshape0*
_output_shapes
:
?
&gradients/logistic_loss/sub_grad/Sum_1Sum5gradients/logistic_loss_grad/tuple/control_dependency8gradients/logistic_loss/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
v
$gradients/logistic_loss/sub_grad/NegNeg&gradients/logistic_loss/sub_grad/Sum_1*
T0*
_output_shapes
:
?
*gradients/logistic_loss/sub_grad/Reshape_1Reshape$gradients/logistic_loss/sub_grad/Neg(gradients/logistic_loss/sub_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
?
1gradients/logistic_loss/sub_grad/tuple/group_depsNoOp)^gradients/logistic_loss/sub_grad/Reshape+^gradients/logistic_loss/sub_grad/Reshape_1
?
9gradients/logistic_loss/sub_grad/tuple/control_dependencyIdentity(gradients/logistic_loss/sub_grad/Reshape2^gradients/logistic_loss/sub_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/logistic_loss/sub_grad/Reshape*
_output_shapes
:
?
;gradients/logistic_loss/sub_grad/tuple/control_dependency_1Identity*gradients/logistic_loss/sub_grad/Reshape_12^gradients/logistic_loss/sub_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/logistic_loss/sub_grad/Reshape_1*
_output_shapes
:
?
(gradients/logistic_loss/Log1p_grad/add/xConst8^gradients/logistic_loss_grad/tuple/control_dependency_1*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
&gradients/logistic_loss/Log1p_grad/addAdd(gradients/logistic_loss/Log1p_grad/add/xlogistic_loss/Exp*
T0*
_output_shapes
:
?
-gradients/logistic_loss/Log1p_grad/Reciprocal
Reciprocal&gradients/logistic_loss/Log1p_grad/add*
T0*
_output_shapes
:
?
&gradients/logistic_loss/Log1p_grad/mulMul7gradients/logistic_loss_grad/tuple/control_dependency_1-gradients/logistic_loss/Log1p_grad/Reciprocal*
T0*
_output_shapes
:
w
0gradients/logistic_loss_1/Select_grad/zeros_like	ZerosLikediscriminator_1/sub_1*
T0*
_output_shapes
:
?
,gradients/logistic_loss_1/Select_grad/SelectSelectlogistic_loss_1/GreaterEqual;gradients/logistic_loss_1/sub_grad/tuple/control_dependency0gradients/logistic_loss_1/Select_grad/zeros_like*
T0*
_output_shapes
:
?
.gradients/logistic_loss_1/Select_grad/Select_1Selectlogistic_loss_1/GreaterEqual0gradients/logistic_loss_1/Select_grad/zeros_like;gradients/logistic_loss_1/sub_grad/tuple/control_dependency*
T0*
_output_shapes
:
?
6gradients/logistic_loss_1/Select_grad/tuple/group_depsNoOp-^gradients/logistic_loss_1/Select_grad/Select/^gradients/logistic_loss_1/Select_grad/Select_1
?
>gradients/logistic_loss_1/Select_grad/tuple/control_dependencyIdentity,gradients/logistic_loss_1/Select_grad/Select7^gradients/logistic_loss_1/Select_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/logistic_loss_1/Select_grad/Select*
_output_shapes
:
?
@gradients/logistic_loss_1/Select_grad/tuple/control_dependency_1Identity.gradients/logistic_loss_1/Select_grad/Select_17^gradients/logistic_loss_1/Select_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/logistic_loss_1/Select_grad/Select_1*
_output_shapes
:
?
(gradients/logistic_loss_1/mul_grad/ShapeShapediscriminator_1/sub_1*
T0*
out_type0*#
_output_shapes
:?????????
}
*gradients/logistic_loss_1/mul_grad/Shape_1Shape
zeros_like*
T0*
out_type0*#
_output_shapes
:?????????
?
8gradients/logistic_loss_1/mul_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients/logistic_loss_1/mul_grad/Shape*gradients/logistic_loss_1/mul_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
&gradients/logistic_loss_1/mul_grad/MulMul=gradients/logistic_loss_1/sub_grad/tuple/control_dependency_1
zeros_like*
T0*
_output_shapes
:
?
&gradients/logistic_loss_1/mul_grad/SumSum&gradients/logistic_loss_1/mul_grad/Mul8gradients/logistic_loss_1/mul_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
?
*gradients/logistic_loss_1/mul_grad/ReshapeReshape&gradients/logistic_loss_1/mul_grad/Sum(gradients/logistic_loss_1/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
:
?
(gradients/logistic_loss_1/mul_grad/Mul_1Muldiscriminator_1/sub_1=gradients/logistic_loss_1/sub_grad/tuple/control_dependency_1*
T0*
_output_shapes
:
?
(gradients/logistic_loss_1/mul_grad/Sum_1Sum(gradients/logistic_loss_1/mul_grad/Mul_1:gradients/logistic_loss_1/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
?
,gradients/logistic_loss_1/mul_grad/Reshape_1Reshape(gradients/logistic_loss_1/mul_grad/Sum_1*gradients/logistic_loss_1/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
?
3gradients/logistic_loss_1/mul_grad/tuple/group_depsNoOp+^gradients/logistic_loss_1/mul_grad/Reshape-^gradients/logistic_loss_1/mul_grad/Reshape_1
?
;gradients/logistic_loss_1/mul_grad/tuple/control_dependencyIdentity*gradients/logistic_loss_1/mul_grad/Reshape4^gradients/logistic_loss_1/mul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/logistic_loss_1/mul_grad/Reshape*
_output_shapes
:
?
=gradients/logistic_loss_1/mul_grad/tuple/control_dependency_1Identity,gradients/logistic_loss_1/mul_grad/Reshape_14^gradients/logistic_loss_1/mul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/logistic_loss_1/mul_grad/Reshape_1*
_output_shapes
:
?
&gradients/logistic_loss_1/Exp_grad/mulMul(gradients/logistic_loss_1/Log1p_grad/mullogistic_loss_1/Exp*
T0*
_output_shapes
:
?
gradients/Squeeze_1_grad/ShapeShape.softmax_cross_entropy_with_logits_sg/Reshape_2*
T0*
out_type0*
_output_shapes
:
?
 gradients/Squeeze_1_grad/ReshapeReshape-gradients/mul_grad/tuple/control_dependency_1gradients/Squeeze_1_grad/Shape*
T0*
Tshape0*#
_output_shapes
:?????????
s
.gradients/logistic_loss/Select_grad/zeros_like	ZerosLikediscriminator/sub_1*
T0*
_output_shapes
:
?
*gradients/logistic_loss/Select_grad/SelectSelectlogistic_loss/GreaterEqual9gradients/logistic_loss/sub_grad/tuple/control_dependency.gradients/logistic_loss/Select_grad/zeros_like*
T0*
_output_shapes
:
?
,gradients/logistic_loss/Select_grad/Select_1Selectlogistic_loss/GreaterEqual.gradients/logistic_loss/Select_grad/zeros_like9gradients/logistic_loss/sub_grad/tuple/control_dependency*
T0*
_output_shapes
:
?
4gradients/logistic_loss/Select_grad/tuple/group_depsNoOp+^gradients/logistic_loss/Select_grad/Select-^gradients/logistic_loss/Select_grad/Select_1
?
<gradients/logistic_loss/Select_grad/tuple/control_dependencyIdentity*gradients/logistic_loss/Select_grad/Select5^gradients/logistic_loss/Select_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/logistic_loss/Select_grad/Select*
_output_shapes
:
?
>gradients/logistic_loss/Select_grad/tuple/control_dependency_1Identity,gradients/logistic_loss/Select_grad/Select_15^gradients/logistic_loss/Select_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/logistic_loss/Select_grad/Select_1*
_output_shapes
:
?
&gradients/logistic_loss/mul_grad/ShapeShapediscriminator/sub_1*
T0*
out_type0*#
_output_shapes
:?????????
z
(gradients/logistic_loss/mul_grad/Shape_1Shape	ones_like*
T0*
out_type0*#
_output_shapes
:?????????
?
6gradients/logistic_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients/logistic_loss/mul_grad/Shape(gradients/logistic_loss/mul_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
$gradients/logistic_loss/mul_grad/MulMul;gradients/logistic_loss/sub_grad/tuple/control_dependency_1	ones_like*
T0*
_output_shapes
:
?
$gradients/logistic_loss/mul_grad/SumSum$gradients/logistic_loss/mul_grad/Mul6gradients/logistic_loss/mul_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
?
(gradients/logistic_loss/mul_grad/ReshapeReshape$gradients/logistic_loss/mul_grad/Sum&gradients/logistic_loss/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
:
?
&gradients/logistic_loss/mul_grad/Mul_1Muldiscriminator/sub_1;gradients/logistic_loss/sub_grad/tuple/control_dependency_1*
T0*
_output_shapes
:
?
&gradients/logistic_loss/mul_grad/Sum_1Sum&gradients/logistic_loss/mul_grad/Mul_18gradients/logistic_loss/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
?
*gradients/logistic_loss/mul_grad/Reshape_1Reshape&gradients/logistic_loss/mul_grad/Sum_1(gradients/logistic_loss/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
?
1gradients/logistic_loss/mul_grad/tuple/group_depsNoOp)^gradients/logistic_loss/mul_grad/Reshape+^gradients/logistic_loss/mul_grad/Reshape_1
?
9gradients/logistic_loss/mul_grad/tuple/control_dependencyIdentity(gradients/logistic_loss/mul_grad/Reshape2^gradients/logistic_loss/mul_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/logistic_loss/mul_grad/Reshape*
_output_shapes
:
?
;gradients/logistic_loss/mul_grad/tuple/control_dependency_1Identity*gradients/logistic_loss/mul_grad/Reshape_12^gradients/logistic_loss/mul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/logistic_loss/mul_grad/Reshape_1*
_output_shapes
:
?
$gradients/logistic_loss/Exp_grad/mulMul&gradients/logistic_loss/Log1p_grad/mullogistic_loss/Exp*
T0*
_output_shapes
:
w
2gradients/logistic_loss_1/Select_1_grad/zeros_like	ZerosLikelogistic_loss_1/Neg*
T0*
_output_shapes
:
?
.gradients/logistic_loss_1/Select_1_grad/SelectSelectlogistic_loss_1/GreaterEqual&gradients/logistic_loss_1/Exp_grad/mul2gradients/logistic_loss_1/Select_1_grad/zeros_like*
T0*
_output_shapes
:
?
0gradients/logistic_loss_1/Select_1_grad/Select_1Selectlogistic_loss_1/GreaterEqual2gradients/logistic_loss_1/Select_1_grad/zeros_like&gradients/logistic_loss_1/Exp_grad/mul*
T0*
_output_shapes
:
?
8gradients/logistic_loss_1/Select_1_grad/tuple/group_depsNoOp/^gradients/logistic_loss_1/Select_1_grad/Select1^gradients/logistic_loss_1/Select_1_grad/Select_1
?
@gradients/logistic_loss_1/Select_1_grad/tuple/control_dependencyIdentity.gradients/logistic_loss_1/Select_1_grad/Select9^gradients/logistic_loss_1/Select_1_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/logistic_loss_1/Select_1_grad/Select*
_output_shapes
:
?
Bgradients/logistic_loss_1/Select_1_grad/tuple/control_dependency_1Identity0gradients/logistic_loss_1/Select_1_grad/Select_19^gradients/logistic_loss_1/Select_1_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/logistic_loss_1/Select_1_grad/Select_1*
_output_shapes
:
?
Cgradients/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ShapeShape$softmax_cross_entropy_with_logits_sg*
T0*
out_type0*
_output_shapes
:
?
Egradients/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeReshape gradients/Squeeze_1_grad/ReshapeCgradients/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/Shape*
T0*
Tshape0*#
_output_shapes
:?????????
s
0gradients/logistic_loss/Select_1_grad/zeros_like	ZerosLikelogistic_loss/Neg*
T0*
_output_shapes
:
?
,gradients/logistic_loss/Select_1_grad/SelectSelectlogistic_loss/GreaterEqual$gradients/logistic_loss/Exp_grad/mul0gradients/logistic_loss/Select_1_grad/zeros_like*
T0*
_output_shapes
:
?
.gradients/logistic_loss/Select_1_grad/Select_1Selectlogistic_loss/GreaterEqual0gradients/logistic_loss/Select_1_grad/zeros_like$gradients/logistic_loss/Exp_grad/mul*
T0*
_output_shapes
:
?
6gradients/logistic_loss/Select_1_grad/tuple/group_depsNoOp-^gradients/logistic_loss/Select_1_grad/Select/^gradients/logistic_loss/Select_1_grad/Select_1
?
>gradients/logistic_loss/Select_1_grad/tuple/control_dependencyIdentity,gradients/logistic_loss/Select_1_grad/Select7^gradients/logistic_loss/Select_1_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/logistic_loss/Select_1_grad/Select*
_output_shapes
:
?
@gradients/logistic_loss/Select_1_grad/tuple/control_dependency_1Identity.gradients/logistic_loss/Select_1_grad/Select_17^gradients/logistic_loss/Select_1_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/logistic_loss/Select_1_grad/Select_1*
_output_shapes
:
?
&gradients/logistic_loss_1/Neg_grad/NegNeg@gradients/logistic_loss_1/Select_1_grad/tuple/control_dependency*
T0*
_output_shapes
:
?
gradients/zeros_like	ZerosLike&softmax_cross_entropy_with_logits_sg:1*
T0*0
_output_shapes
:??????????????????
?
Bgradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims/dimConst*
valueB :
?????????*
dtype0*
_output_shapes
: 
?
>gradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims
ExpandDimsEgradients/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeBgradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims/dim*
T0*

Tdim0*'
_output_shapes
:?????????
?
7gradients/softmax_cross_entropy_with_logits_sg_grad/mulMul>gradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims&softmax_cross_entropy_with_logits_sg:1*
T0*0
_output_shapes
:??????????????????
?
>gradients/softmax_cross_entropy_with_logits_sg_grad/LogSoftmax
LogSoftmax,softmax_cross_entropy_with_logits_sg/Reshape*
T0*0
_output_shapes
:??????????????????
?
7gradients/softmax_cross_entropy_with_logits_sg_grad/NegNeg>gradients/softmax_cross_entropy_with_logits_sg_grad/LogSoftmax*
T0*0
_output_shapes
:??????????????????
?
Dgradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1/dimConst*
valueB :
?????????*
dtype0*
_output_shapes
: 
?
@gradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1
ExpandDimsEgradients/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeDgradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1/dim*
T0*

Tdim0*'
_output_shapes
:?????????
?
9gradients/softmax_cross_entropy_with_logits_sg_grad/mul_1Mul@gradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_17gradients/softmax_cross_entropy_with_logits_sg_grad/Neg*
T0*0
_output_shapes
:??????????????????
?
Dgradients/softmax_cross_entropy_with_logits_sg_grad/tuple/group_depsNoOp8^gradients/softmax_cross_entropy_with_logits_sg_grad/mul:^gradients/softmax_cross_entropy_with_logits_sg_grad/mul_1
?
Lgradients/softmax_cross_entropy_with_logits_sg_grad/tuple/control_dependencyIdentity7gradients/softmax_cross_entropy_with_logits_sg_grad/mulE^gradients/softmax_cross_entropy_with_logits_sg_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/softmax_cross_entropy_with_logits_sg_grad/mul*0
_output_shapes
:??????????????????
?
Ngradients/softmax_cross_entropy_with_logits_sg_grad/tuple/control_dependency_1Identity9gradients/softmax_cross_entropy_with_logits_sg_grad/mul_1E^gradients/softmax_cross_entropy_with_logits_sg_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/softmax_cross_entropy_with_logits_sg_grad/mul_1*0
_output_shapes
:??????????????????
?
$gradients/logistic_loss/Neg_grad/NegNeg>gradients/logistic_loss/Select_1_grad/tuple/control_dependency*
T0*
_output_shapes
:
?
Agradients/softmax_cross_entropy_with_logits_sg/Reshape_grad/ShapeShapediscriminator/dense/BiasAdd*
T0*
out_type0*
_output_shapes
:
?
Cgradients/softmax_cross_entropy_with_logits_sg/Reshape_grad/ReshapeReshapeLgradients/softmax_cross_entropy_with_logits_sg_grad/tuple/control_dependencyAgradients/softmax_cross_entropy_with_logits_sg/Reshape_grad/Shape*
T0*
Tshape0*'
_output_shapes
:??????????
?
gradients/AddNAddN>gradients/logistic_loss_1/Select_grad/tuple/control_dependency;gradients/logistic_loss_1/mul_grad/tuple/control_dependencyBgradients/logistic_loss_1/Select_1_grad/tuple/control_dependency_1&gradients/logistic_loss_1/Neg_grad/Neg*
T0*
N*?
_class5
31loc:@gradients/logistic_loss_1/Select_grad/Select*
_output_shapes
:
?
*gradients/discriminator_1/sub_1_grad/ShapeShapediscriminator_1/add*
T0*
out_type0*#
_output_shapes
:?????????
o
,gradients/discriminator_1/sub_1_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
?
:gradients/discriminator_1/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients/discriminator_1/sub_1_grad/Shape,gradients/discriminator_1/sub_1_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
(gradients/discriminator_1/sub_1_grad/SumSumgradients/AddN:gradients/discriminator_1/sub_1_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
?
,gradients/discriminator_1/sub_1_grad/ReshapeReshape(gradients/discriminator_1/sub_1_grad/Sum*gradients/discriminator_1/sub_1_grad/Shape*
T0*
Tshape0*
_output_shapes
:
?
*gradients/discriminator_1/sub_1_grad/Sum_1Sumgradients/AddN<gradients/discriminator_1/sub_1_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
~
(gradients/discriminator_1/sub_1_grad/NegNeg*gradients/discriminator_1/sub_1_grad/Sum_1*
T0*
_output_shapes
:
?
.gradients/discriminator_1/sub_1_grad/Reshape_1Reshape(gradients/discriminator_1/sub_1_grad/Neg,gradients/discriminator_1/sub_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
?
5gradients/discriminator_1/sub_1_grad/tuple/group_depsNoOp-^gradients/discriminator_1/sub_1_grad/Reshape/^gradients/discriminator_1/sub_1_grad/Reshape_1
?
=gradients/discriminator_1/sub_1_grad/tuple/control_dependencyIdentity,gradients/discriminator_1/sub_1_grad/Reshape6^gradients/discriminator_1/sub_1_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/discriminator_1/sub_1_grad/Reshape*
_output_shapes
:
?
?gradients/discriminator_1/sub_1_grad/tuple/control_dependency_1Identity.gradients/discriminator_1/sub_1_grad/Reshape_16^gradients/discriminator_1/sub_1_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/discriminator_1/sub_1_grad/Reshape_1*
_output_shapes
: 
?
gradients/AddN_1AddN<gradients/logistic_loss/Select_grad/tuple/control_dependency9gradients/logistic_loss/mul_grad/tuple/control_dependency@gradients/logistic_loss/Select_1_grad/tuple/control_dependency_1$gradients/logistic_loss/Neg_grad/Neg*
T0*
N*=
_class3
1/loc:@gradients/logistic_loss/Select_grad/Select*
_output_shapes
:
?
(gradients/discriminator/sub_1_grad/ShapeShapediscriminator/add*
T0*
out_type0*#
_output_shapes
:?????????
m
*gradients/discriminator/sub_1_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
?
8gradients/discriminator/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients/discriminator/sub_1_grad/Shape*gradients/discriminator/sub_1_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
&gradients/discriminator/sub_1_grad/SumSumgradients/AddN_18gradients/discriminator/sub_1_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
?
*gradients/discriminator/sub_1_grad/ReshapeReshape&gradients/discriminator/sub_1_grad/Sum(gradients/discriminator/sub_1_grad/Shape*
T0*
Tshape0*
_output_shapes
:
?
(gradients/discriminator/sub_1_grad/Sum_1Sumgradients/AddN_1:gradients/discriminator/sub_1_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
z
&gradients/discriminator/sub_1_grad/NegNeg(gradients/discriminator/sub_1_grad/Sum_1*
T0*
_output_shapes
:
?
,gradients/discriminator/sub_1_grad/Reshape_1Reshape&gradients/discriminator/sub_1_grad/Neg*gradients/discriminator/sub_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
?
3gradients/discriminator/sub_1_grad/tuple/group_depsNoOp+^gradients/discriminator/sub_1_grad/Reshape-^gradients/discriminator/sub_1_grad/Reshape_1
?
;gradients/discriminator/sub_1_grad/tuple/control_dependencyIdentity*gradients/discriminator/sub_1_grad/Reshape4^gradients/discriminator/sub_1_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/discriminator/sub_1_grad/Reshape*
_output_shapes
:
?
=gradients/discriminator/sub_1_grad/tuple/control_dependency_1Identity,gradients/discriminator/sub_1_grad/Reshape_14^gradients/discriminator/sub_1_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/discriminator/sub_1_grad/Reshape_1*
_output_shapes
: 
{
(gradients/discriminator_1/add_grad/ShapeShapediscriminator_1/Log*
T0*
out_type0*
_output_shapes
:
?
*gradients/discriminator_1/add_grad/Shape_1Shapediscriminator_1/Squeeze*
T0*
out_type0*#
_output_shapes
:?????????
?
8gradients/discriminator_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients/discriminator_1/add_grad/Shape*gradients/discriminator_1/add_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
&gradients/discriminator_1/add_grad/SumSum=gradients/discriminator_1/sub_1_grad/tuple/control_dependency8gradients/discriminator_1/add_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
?
*gradients/discriminator_1/add_grad/ReshapeReshape&gradients/discriminator_1/add_grad/Sum(gradients/discriminator_1/add_grad/Shape*
T0*
Tshape0*#
_output_shapes
:?????????
?
(gradients/discriminator_1/add_grad/Sum_1Sum=gradients/discriminator_1/sub_1_grad/tuple/control_dependency:gradients/discriminator_1/add_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
?
,gradients/discriminator_1/add_grad/Reshape_1Reshape(gradients/discriminator_1/add_grad/Sum_1*gradients/discriminator_1/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
?
3gradients/discriminator_1/add_grad/tuple/group_depsNoOp+^gradients/discriminator_1/add_grad/Reshape-^gradients/discriminator_1/add_grad/Reshape_1
?
;gradients/discriminator_1/add_grad/tuple/control_dependencyIdentity*gradients/discriminator_1/add_grad/Reshape4^gradients/discriminator_1/add_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/discriminator_1/add_grad/Reshape*#
_output_shapes
:?????????
?
=gradients/discriminator_1/add_grad/tuple/control_dependency_1Identity,gradients/discriminator_1/add_grad/Reshape_14^gradients/discriminator_1/add_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/discriminator_1/add_grad/Reshape_1*
_output_shapes
:
w
&gradients/discriminator/add_grad/ShapeShapediscriminator/Log*
T0*
out_type0*
_output_shapes
:
?
(gradients/discriminator/add_grad/Shape_1Shapediscriminator/Squeeze*
T0*
out_type0*#
_output_shapes
:?????????
?
6gradients/discriminator/add_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients/discriminator/add_grad/Shape(gradients/discriminator/add_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
$gradients/discriminator/add_grad/SumSum;gradients/discriminator/sub_1_grad/tuple/control_dependency6gradients/discriminator/add_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
?
(gradients/discriminator/add_grad/ReshapeReshape$gradients/discriminator/add_grad/Sum&gradients/discriminator/add_grad/Shape*
T0*
Tshape0*#
_output_shapes
:?????????
?
&gradients/discriminator/add_grad/Sum_1Sum;gradients/discriminator/sub_1_grad/tuple/control_dependency8gradients/discriminator/add_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
?
*gradients/discriminator/add_grad/Reshape_1Reshape&gradients/discriminator/add_grad/Sum_1(gradients/discriminator/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
?
1gradients/discriminator/add_grad/tuple/group_depsNoOp)^gradients/discriminator/add_grad/Reshape+^gradients/discriminator/add_grad/Reshape_1
?
9gradients/discriminator/add_grad/tuple/control_dependencyIdentity(gradients/discriminator/add_grad/Reshape2^gradients/discriminator/add_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/discriminator/add_grad/Reshape*#
_output_shapes
:?????????
?
;gradients/discriminator/add_grad/tuple/control_dependency_1Identity*gradients/discriminator/add_grad/Reshape_12^gradients/discriminator/add_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/discriminator/add_grad/Reshape_1*
_output_shapes
:
?
-gradients/discriminator_1/Log_grad/Reciprocal
Reciprocaldiscriminator_1/Sum<^gradients/discriminator_1/add_grad/tuple/control_dependency*
T0*#
_output_shapes
:?????????
?
&gradients/discriminator_1/Log_grad/mulMul;gradients/discriminator_1/add_grad/tuple/control_dependency-gradients/discriminator_1/Log_grad/Reciprocal*
T0*#
_output_shapes
:?????????

,gradients/discriminator_1/Squeeze_grad/ShapeShapediscriminator_1/Max*
T0*
out_type0*
_output_shapes
:
?
.gradients/discriminator_1/Squeeze_grad/ReshapeReshape=gradients/discriminator_1/add_grad/tuple/control_dependency_1,gradients/discriminator_1/Squeeze_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
+gradients/discriminator/Log_grad/Reciprocal
Reciprocaldiscriminator/Sum:^gradients/discriminator/add_grad/tuple/control_dependency*
T0*#
_output_shapes
:?????????
?
$gradients/discriminator/Log_grad/mulMul9gradients/discriminator/add_grad/tuple/control_dependency+gradients/discriminator/Log_grad/Reciprocal*
T0*#
_output_shapes
:?????????
{
*gradients/discriminator/Squeeze_grad/ShapeShapediscriminator/Max*
T0*
out_type0*
_output_shapes
:
?
,gradients/discriminator/Squeeze_grad/ReshapeReshape;gradients/discriminator/add_grad/tuple/control_dependency_1*gradients/discriminator/Squeeze_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
{
(gradients/discriminator_1/Sum_grad/ShapeShapediscriminator_1/Exp*
T0*
out_type0*
_output_shapes
:
?
'gradients/discriminator_1/Sum_grad/SizeConst*
value	B :*
dtype0*;
_class1
/-loc:@gradients/discriminator_1/Sum_grad/Shape*
_output_shapes
: 
?
&gradients/discriminator_1/Sum_grad/addAdd%discriminator_1/Sum/reduction_indices'gradients/discriminator_1/Sum_grad/Size*
T0*;
_class1
/-loc:@gradients/discriminator_1/Sum_grad/Shape*
_output_shapes
: 
?
&gradients/discriminator_1/Sum_grad/modFloorMod&gradients/discriminator_1/Sum_grad/add'gradients/discriminator_1/Sum_grad/Size*
T0*;
_class1
/-loc:@gradients/discriminator_1/Sum_grad/Shape*
_output_shapes
: 
?
*gradients/discriminator_1/Sum_grad/Shape_1Const*
valueB *
dtype0*;
_class1
/-loc:@gradients/discriminator_1/Sum_grad/Shape*
_output_shapes
: 
?
.gradients/discriminator_1/Sum_grad/range/startConst*
value	B : *
dtype0*;
_class1
/-loc:@gradients/discriminator_1/Sum_grad/Shape*
_output_shapes
: 
?
.gradients/discriminator_1/Sum_grad/range/deltaConst*
value	B :*
dtype0*;
_class1
/-loc:@gradients/discriminator_1/Sum_grad/Shape*
_output_shapes
: 
?
(gradients/discriminator_1/Sum_grad/rangeRange.gradients/discriminator_1/Sum_grad/range/start'gradients/discriminator_1/Sum_grad/Size.gradients/discriminator_1/Sum_grad/range/delta*

Tidx0*;
_class1
/-loc:@gradients/discriminator_1/Sum_grad/Shape*
_output_shapes
:
?
-gradients/discriminator_1/Sum_grad/Fill/valueConst*
value	B :*
dtype0*;
_class1
/-loc:@gradients/discriminator_1/Sum_grad/Shape*
_output_shapes
: 
?
'gradients/discriminator_1/Sum_grad/FillFill*gradients/discriminator_1/Sum_grad/Shape_1-gradients/discriminator_1/Sum_grad/Fill/value*
T0*

index_type0*;
_class1
/-loc:@gradients/discriminator_1/Sum_grad/Shape*
_output_shapes
: 
?
0gradients/discriminator_1/Sum_grad/DynamicStitchDynamicStitch(gradients/discriminator_1/Sum_grad/range&gradients/discriminator_1/Sum_grad/mod(gradients/discriminator_1/Sum_grad/Shape'gradients/discriminator_1/Sum_grad/Fill*
T0*
N*;
_class1
/-loc:@gradients/discriminator_1/Sum_grad/Shape*
_output_shapes
:
?
,gradients/discriminator_1/Sum_grad/Maximum/yConst*
value	B :*
dtype0*;
_class1
/-loc:@gradients/discriminator_1/Sum_grad/Shape*
_output_shapes
: 
?
*gradients/discriminator_1/Sum_grad/MaximumMaximum0gradients/discriminator_1/Sum_grad/DynamicStitch,gradients/discriminator_1/Sum_grad/Maximum/y*
T0*;
_class1
/-loc:@gradients/discriminator_1/Sum_grad/Shape*
_output_shapes
:
?
+gradients/discriminator_1/Sum_grad/floordivFloorDiv(gradients/discriminator_1/Sum_grad/Shape*gradients/discriminator_1/Sum_grad/Maximum*
T0*;
_class1
/-loc:@gradients/discriminator_1/Sum_grad/Shape*
_output_shapes
:
?
*gradients/discriminator_1/Sum_grad/ReshapeReshape&gradients/discriminator_1/Log_grad/mul0gradients/discriminator_1/Sum_grad/DynamicStitch*
T0*
Tshape0*0
_output_shapes
:??????????????????
?
'gradients/discriminator_1/Sum_grad/TileTile*gradients/discriminator_1/Sum_grad/Reshape+gradients/discriminator_1/Sum_grad/floordiv*
T0*

Tmultiples0*'
_output_shapes
:??????????
w
&gradients/discriminator/Sum_grad/ShapeShapediscriminator/Exp*
T0*
out_type0*
_output_shapes
:
?
%gradients/discriminator/Sum_grad/SizeConst*
value	B :*
dtype0*9
_class/
-+loc:@gradients/discriminator/Sum_grad/Shape*
_output_shapes
: 
?
$gradients/discriminator/Sum_grad/addAdd#discriminator/Sum/reduction_indices%gradients/discriminator/Sum_grad/Size*
T0*9
_class/
-+loc:@gradients/discriminator/Sum_grad/Shape*
_output_shapes
: 
?
$gradients/discriminator/Sum_grad/modFloorMod$gradients/discriminator/Sum_grad/add%gradients/discriminator/Sum_grad/Size*
T0*9
_class/
-+loc:@gradients/discriminator/Sum_grad/Shape*
_output_shapes
: 
?
(gradients/discriminator/Sum_grad/Shape_1Const*
valueB *
dtype0*9
_class/
-+loc:@gradients/discriminator/Sum_grad/Shape*
_output_shapes
: 
?
,gradients/discriminator/Sum_grad/range/startConst*
value	B : *
dtype0*9
_class/
-+loc:@gradients/discriminator/Sum_grad/Shape*
_output_shapes
: 
?
,gradients/discriminator/Sum_grad/range/deltaConst*
value	B :*
dtype0*9
_class/
-+loc:@gradients/discriminator/Sum_grad/Shape*
_output_shapes
: 
?
&gradients/discriminator/Sum_grad/rangeRange,gradients/discriminator/Sum_grad/range/start%gradients/discriminator/Sum_grad/Size,gradients/discriminator/Sum_grad/range/delta*

Tidx0*9
_class/
-+loc:@gradients/discriminator/Sum_grad/Shape*
_output_shapes
:
?
+gradients/discriminator/Sum_grad/Fill/valueConst*
value	B :*
dtype0*9
_class/
-+loc:@gradients/discriminator/Sum_grad/Shape*
_output_shapes
: 
?
%gradients/discriminator/Sum_grad/FillFill(gradients/discriminator/Sum_grad/Shape_1+gradients/discriminator/Sum_grad/Fill/value*
T0*

index_type0*9
_class/
-+loc:@gradients/discriminator/Sum_grad/Shape*
_output_shapes
: 
?
.gradients/discriminator/Sum_grad/DynamicStitchDynamicStitch&gradients/discriminator/Sum_grad/range$gradients/discriminator/Sum_grad/mod&gradients/discriminator/Sum_grad/Shape%gradients/discriminator/Sum_grad/Fill*
T0*
N*9
_class/
-+loc:@gradients/discriminator/Sum_grad/Shape*
_output_shapes
:
?
*gradients/discriminator/Sum_grad/Maximum/yConst*
value	B :*
dtype0*9
_class/
-+loc:@gradients/discriminator/Sum_grad/Shape*
_output_shapes
: 
?
(gradients/discriminator/Sum_grad/MaximumMaximum.gradients/discriminator/Sum_grad/DynamicStitch*gradients/discriminator/Sum_grad/Maximum/y*
T0*9
_class/
-+loc:@gradients/discriminator/Sum_grad/Shape*
_output_shapes
:
?
)gradients/discriminator/Sum_grad/floordivFloorDiv&gradients/discriminator/Sum_grad/Shape(gradients/discriminator/Sum_grad/Maximum*
T0*9
_class/
-+loc:@gradients/discriminator/Sum_grad/Shape*
_output_shapes
:
?
(gradients/discriminator/Sum_grad/ReshapeReshape$gradients/discriminator/Log_grad/mul.gradients/discriminator/Sum_grad/DynamicStitch*
T0*
Tshape0*0
_output_shapes
:??????????????????
?
%gradients/discriminator/Sum_grad/TileTile(gradients/discriminator/Sum_grad/Reshape)gradients/discriminator/Sum_grad/floordiv*
T0*

Tmultiples0*'
_output_shapes
:??????????
?
&gradients/discriminator_1/Exp_grad/mulMul'gradients/discriminator_1/Sum_grad/Tilediscriminator_1/Exp*
T0*'
_output_shapes
:??????????
?
$gradients/discriminator/Exp_grad/mulMul%gradients/discriminator/Sum_grad/Tilediscriminator/Exp*
T0*'
_output_shapes
:??????????
?
(gradients/discriminator_1/sub_grad/ShapeShapediscriminator_1/dense/BiasAdd*
T0*
out_type0*
_output_shapes
:
}
*gradients/discriminator_1/sub_grad/Shape_1Shapediscriminator_1/Max*
T0*
out_type0*
_output_shapes
:
?
8gradients/discriminator_1/sub_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients/discriminator_1/sub_grad/Shape*gradients/discriminator_1/sub_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
&gradients/discriminator_1/sub_grad/SumSum&gradients/discriminator_1/Exp_grad/mul8gradients/discriminator_1/sub_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
?
*gradients/discriminator_1/sub_grad/ReshapeReshape&gradients/discriminator_1/sub_grad/Sum(gradients/discriminator_1/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:??????????
?
(gradients/discriminator_1/sub_grad/Sum_1Sum&gradients/discriminator_1/Exp_grad/mul:gradients/discriminator_1/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
z
&gradients/discriminator_1/sub_grad/NegNeg(gradients/discriminator_1/sub_grad/Sum_1*
T0*
_output_shapes
:
?
,gradients/discriminator_1/sub_grad/Reshape_1Reshape&gradients/discriminator_1/sub_grad/Neg*gradients/discriminator_1/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:?????????
?
3gradients/discriminator_1/sub_grad/tuple/group_depsNoOp+^gradients/discriminator_1/sub_grad/Reshape-^gradients/discriminator_1/sub_grad/Reshape_1
?
;gradients/discriminator_1/sub_grad/tuple/control_dependencyIdentity*gradients/discriminator_1/sub_grad/Reshape4^gradients/discriminator_1/sub_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/discriminator_1/sub_grad/Reshape*'
_output_shapes
:??????????
?
=gradients/discriminator_1/sub_grad/tuple/control_dependency_1Identity,gradients/discriminator_1/sub_grad/Reshape_14^gradients/discriminator_1/sub_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/discriminator_1/sub_grad/Reshape_1*'
_output_shapes
:?????????
?
&gradients/discriminator/sub_grad/ShapeShapediscriminator/dense/BiasAdd*
T0*
out_type0*
_output_shapes
:
y
(gradients/discriminator/sub_grad/Shape_1Shapediscriminator/Max*
T0*
out_type0*
_output_shapes
:
?
6gradients/discriminator/sub_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients/discriminator/sub_grad/Shape(gradients/discriminator/sub_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
$gradients/discriminator/sub_grad/SumSum$gradients/discriminator/Exp_grad/mul6gradients/discriminator/sub_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
?
(gradients/discriminator/sub_grad/ReshapeReshape$gradients/discriminator/sub_grad/Sum&gradients/discriminator/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:??????????
?
&gradients/discriminator/sub_grad/Sum_1Sum$gradients/discriminator/Exp_grad/mul8gradients/discriminator/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
v
$gradients/discriminator/sub_grad/NegNeg&gradients/discriminator/sub_grad/Sum_1*
T0*
_output_shapes
:
?
*gradients/discriminator/sub_grad/Reshape_1Reshape$gradients/discriminator/sub_grad/Neg(gradients/discriminator/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:?????????
?
1gradients/discriminator/sub_grad/tuple/group_depsNoOp)^gradients/discriminator/sub_grad/Reshape+^gradients/discriminator/sub_grad/Reshape_1
?
9gradients/discriminator/sub_grad/tuple/control_dependencyIdentity(gradients/discriminator/sub_grad/Reshape2^gradients/discriminator/sub_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/discriminator/sub_grad/Reshape*'
_output_shapes
:??????????
?
;gradients/discriminator/sub_grad/tuple/control_dependency_1Identity*gradients/discriminator/sub_grad/Reshape_12^gradients/discriminator/sub_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/discriminator/sub_grad/Reshape_1*'
_output_shapes
:?????????
?
gradients/AddN_2AddN.gradients/discriminator_1/Squeeze_grad/Reshape=gradients/discriminator_1/sub_grad/tuple/control_dependency_1*
T0*
N*A
_class7
53loc:@gradients/discriminator_1/Squeeze_grad/Reshape*'
_output_shapes
:?????????
?
(gradients/discriminator_1/Max_grad/ShapeShapediscriminator_1/dense/BiasAdd*
T0*
out_type0*
_output_shapes
:
i
'gradients/discriminator_1/Max_grad/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
?
&gradients/discriminator_1/Max_grad/addAdd%discriminator_1/Max/reduction_indices'gradients/discriminator_1/Max_grad/Size*
T0*
_output_shapes
: 
?
&gradients/discriminator_1/Max_grad/modFloorMod&gradients/discriminator_1/Max_grad/add'gradients/discriminator_1/Max_grad/Size*
T0*
_output_shapes
: 
m
*gradients/discriminator_1/Max_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
p
.gradients/discriminator_1/Max_grad/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
p
.gradients/discriminator_1/Max_grad/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
?
(gradients/discriminator_1/Max_grad/rangeRange.gradients/discriminator_1/Max_grad/range/start'gradients/discriminator_1/Max_grad/Size.gradients/discriminator_1/Max_grad/range/delta*

Tidx0*
_output_shapes
:
o
-gradients/discriminator_1/Max_grad/Fill/valueConst*
value	B :*
dtype0*
_output_shapes
: 
?
'gradients/discriminator_1/Max_grad/FillFill*gradients/discriminator_1/Max_grad/Shape_1-gradients/discriminator_1/Max_grad/Fill/value*
T0*

index_type0*
_output_shapes
: 
?
0gradients/discriminator_1/Max_grad/DynamicStitchDynamicStitch(gradients/discriminator_1/Max_grad/range&gradients/discriminator_1/Max_grad/mod(gradients/discriminator_1/Max_grad/Shape'gradients/discriminator_1/Max_grad/Fill*
T0*
N*
_output_shapes
:
?
*gradients/discriminator_1/Max_grad/ReshapeReshapediscriminator_1/Max0gradients/discriminator_1/Max_grad/DynamicStitch*
T0*
Tshape0*0
_output_shapes
:??????????????????
?
,gradients/discriminator_1/Max_grad/Reshape_1Reshapegradients/AddN_20gradients/discriminator_1/Max_grad/DynamicStitch*
T0*
Tshape0*0
_output_shapes
:??????????????????
?
(gradients/discriminator_1/Max_grad/EqualEqual*gradients/discriminator_1/Max_grad/Reshapediscriminator_1/dense/BiasAdd*
T0*'
_output_shapes
:??????????
?
'gradients/discriminator_1/Max_grad/CastCast(gradients/discriminator_1/Max_grad/Equal*

DstT0*
Truncate( *

SrcT0
*'
_output_shapes
:??????????
?
&gradients/discriminator_1/Max_grad/SumSum'gradients/discriminator_1/Max_grad/Cast%discriminator_1/Max/reduction_indices*
	keep_dims( *
T0*

Tidx0*#
_output_shapes
:?????????
?
,gradients/discriminator_1/Max_grad/Reshape_2Reshape&gradients/discriminator_1/Max_grad/Sum0gradients/discriminator_1/Max_grad/DynamicStitch*
T0*
Tshape0*0
_output_shapes
:??????????????????
?
&gradients/discriminator_1/Max_grad/divRealDiv'gradients/discriminator_1/Max_grad/Cast,gradients/discriminator_1/Max_grad/Reshape_2*
T0*'
_output_shapes
:??????????
?
&gradients/discriminator_1/Max_grad/mulMul&gradients/discriminator_1/Max_grad/div,gradients/discriminator_1/Max_grad/Reshape_1*
T0*'
_output_shapes
:??????????
?
gradients/AddN_3AddN,gradients/discriminator/Squeeze_grad/Reshape;gradients/discriminator/sub_grad/tuple/control_dependency_1*
T0*
N*?
_class5
31loc:@gradients/discriminator/Squeeze_grad/Reshape*'
_output_shapes
:?????????
?
&gradients/discriminator/Max_grad/ShapeShapediscriminator/dense/BiasAdd*
T0*
out_type0*
_output_shapes
:
g
%gradients/discriminator/Max_grad/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
?
$gradients/discriminator/Max_grad/addAdd#discriminator/Max/reduction_indices%gradients/discriminator/Max_grad/Size*
T0*
_output_shapes
: 
?
$gradients/discriminator/Max_grad/modFloorMod$gradients/discriminator/Max_grad/add%gradients/discriminator/Max_grad/Size*
T0*
_output_shapes
: 
k
(gradients/discriminator/Max_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
n
,gradients/discriminator/Max_grad/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
n
,gradients/discriminator/Max_grad/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
?
&gradients/discriminator/Max_grad/rangeRange,gradients/discriminator/Max_grad/range/start%gradients/discriminator/Max_grad/Size,gradients/discriminator/Max_grad/range/delta*

Tidx0*
_output_shapes
:
m
+gradients/discriminator/Max_grad/Fill/valueConst*
value	B :*
dtype0*
_output_shapes
: 
?
%gradients/discriminator/Max_grad/FillFill(gradients/discriminator/Max_grad/Shape_1+gradients/discriminator/Max_grad/Fill/value*
T0*

index_type0*
_output_shapes
: 
?
.gradients/discriminator/Max_grad/DynamicStitchDynamicStitch&gradients/discriminator/Max_grad/range$gradients/discriminator/Max_grad/mod&gradients/discriminator/Max_grad/Shape%gradients/discriminator/Max_grad/Fill*
T0*
N*
_output_shapes
:
?
(gradients/discriminator/Max_grad/ReshapeReshapediscriminator/Max.gradients/discriminator/Max_grad/DynamicStitch*
T0*
Tshape0*0
_output_shapes
:??????????????????
?
*gradients/discriminator/Max_grad/Reshape_1Reshapegradients/AddN_3.gradients/discriminator/Max_grad/DynamicStitch*
T0*
Tshape0*0
_output_shapes
:??????????????????
?
&gradients/discriminator/Max_grad/EqualEqual(gradients/discriminator/Max_grad/Reshapediscriminator/dense/BiasAdd*
T0*'
_output_shapes
:??????????
?
%gradients/discriminator/Max_grad/CastCast&gradients/discriminator/Max_grad/Equal*

DstT0*
Truncate( *

SrcT0
*'
_output_shapes
:??????????
?
$gradients/discriminator/Max_grad/SumSum%gradients/discriminator/Max_grad/Cast#discriminator/Max/reduction_indices*
	keep_dims( *
T0*

Tidx0*#
_output_shapes
:?????????
?
*gradients/discriminator/Max_grad/Reshape_2Reshape$gradients/discriminator/Max_grad/Sum.gradients/discriminator/Max_grad/DynamicStitch*
T0*
Tshape0*0
_output_shapes
:??????????????????
?
$gradients/discriminator/Max_grad/divRealDiv%gradients/discriminator/Max_grad/Cast*gradients/discriminator/Max_grad/Reshape_2*
T0*'
_output_shapes
:??????????
?
$gradients/discriminator/Max_grad/mulMul$gradients/discriminator/Max_grad/div*gradients/discriminator/Max_grad/Reshape_1*
T0*'
_output_shapes
:??????????
?
gradients/AddN_4AddN;gradients/discriminator_1/sub_grad/tuple/control_dependency&gradients/discriminator_1/Max_grad/mul*
T0*
N*=
_class3
1/loc:@gradients/discriminator_1/sub_grad/Reshape*'
_output_shapes
:??????????
?
8gradients/discriminator_1/dense/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_4*
T0*
data_formatNHWC*
_output_shapes
:?
?
=gradients/discriminator_1/dense/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_49^gradients/discriminator_1/dense/BiasAdd_grad/BiasAddGrad
?
Egradients/discriminator_1/dense/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_4>^gradients/discriminator_1/dense/BiasAdd_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/discriminator_1/sub_grad/Reshape*'
_output_shapes
:??????????
?
Ggradients/discriminator_1/dense/BiasAdd_grad/tuple/control_dependency_1Identity8gradients/discriminator_1/dense/BiasAdd_grad/BiasAddGrad>^gradients/discriminator_1/dense/BiasAdd_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/discriminator_1/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
:?
?
gradients/AddN_5AddNCgradients/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape9gradients/discriminator/sub_grad/tuple/control_dependency$gradients/discriminator/Max_grad/mul*
T0*
N*V
_classL
JHloc:@gradients/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape*'
_output_shapes
:??????????
?
6gradients/discriminator/dense/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_5*
T0*
data_formatNHWC*
_output_shapes
:?
?
;gradients/discriminator/dense/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_57^gradients/discriminator/dense/BiasAdd_grad/BiasAddGrad
?
Cgradients/discriminator/dense/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_5<^gradients/discriminator/dense/BiasAdd_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape*'
_output_shapes
:??????????
?
Egradients/discriminator/dense/BiasAdd_grad/tuple/control_dependency_1Identity6gradients/discriminator/dense/BiasAdd_grad/BiasAddGrad<^gradients/discriminator/dense/BiasAdd_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/discriminator/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
:?
?
2gradients/discriminator_1/dense/MatMul_grad/MatMulMatMulEgradients/discriminator_1/dense/BiasAdd_grad/tuple/control_dependencydiscriminator/dense/kernel/read*
T0*
transpose_b(*
transpose_a( *(
_output_shapes
:??????????
?
4gradients/discriminator_1/dense/MatMul_grad/MatMul_1MatMuldiscriminator_1/MeanEgradients/discriminator_1/dense/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_b( *
transpose_a(*
_output_shapes
:	??
?
<gradients/discriminator_1/dense/MatMul_grad/tuple/group_depsNoOp3^gradients/discriminator_1/dense/MatMul_grad/MatMul5^gradients/discriminator_1/dense/MatMul_grad/MatMul_1
?
Dgradients/discriminator_1/dense/MatMul_grad/tuple/control_dependencyIdentity2gradients/discriminator_1/dense/MatMul_grad/MatMul=^gradients/discriminator_1/dense/MatMul_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/discriminator_1/dense/MatMul_grad/MatMul*(
_output_shapes
:??????????
?
Fgradients/discriminator_1/dense/MatMul_grad/tuple/control_dependency_1Identity4gradients/discriminator_1/dense/MatMul_grad/MatMul_1=^gradients/discriminator_1/dense/MatMul_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/discriminator_1/dense/MatMul_grad/MatMul_1*
_output_shapes
:	??
?
0gradients/discriminator/dense/MatMul_grad/MatMulMatMulCgradients/discriminator/dense/BiasAdd_grad/tuple/control_dependencydiscriminator/dense/kernel/read*
T0*
transpose_b(*
transpose_a( *(
_output_shapes
:??????????
?
2gradients/discriminator/dense/MatMul_grad/MatMul_1MatMuldiscriminator/MeanCgradients/discriminator/dense/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_b( *
transpose_a(*
_output_shapes
:	??
?
:gradients/discriminator/dense/MatMul_grad/tuple/group_depsNoOp1^gradients/discriminator/dense/MatMul_grad/MatMul3^gradients/discriminator/dense/MatMul_grad/MatMul_1
?
Bgradients/discriminator/dense/MatMul_grad/tuple/control_dependencyIdentity0gradients/discriminator/dense/MatMul_grad/MatMul;^gradients/discriminator/dense/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/discriminator/dense/MatMul_grad/MatMul*(
_output_shapes
:??????????
?
Dgradients/discriminator/dense/MatMul_grad/tuple/control_dependency_1Identity2gradients/discriminator/dense/MatMul_grad/MatMul_1;^gradients/discriminator/dense/MatMul_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/discriminator/dense/MatMul_grad/MatMul_1*
_output_shapes
:	??
?
gradients/AddN_6AddNGgradients/discriminator_1/dense/BiasAdd_grad/tuple/control_dependency_1Egradients/discriminator/dense/BiasAdd_grad/tuple/control_dependency_1*
T0*
N*K
_classA
?=loc:@gradients/discriminator_1/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
:?
?
)gradients/discriminator_1/Mean_grad/ShapeShapediscriminator_1/conv2d_5/Relu*
T0*
out_type0*
_output_shapes
:
?
(gradients/discriminator_1/Mean_grad/SizeConst*
value	B :*
dtype0*<
_class2
0.loc:@gradients/discriminator_1/Mean_grad/Shape*
_output_shapes
: 
?
'gradients/discriminator_1/Mean_grad/addAdd&discriminator_1/Mean/reduction_indices(gradients/discriminator_1/Mean_grad/Size*
T0*<
_class2
0.loc:@gradients/discriminator_1/Mean_grad/Shape*
_output_shapes
:
?
'gradients/discriminator_1/Mean_grad/modFloorMod'gradients/discriminator_1/Mean_grad/add(gradients/discriminator_1/Mean_grad/Size*
T0*<
_class2
0.loc:@gradients/discriminator_1/Mean_grad/Shape*
_output_shapes
:
?
+gradients/discriminator_1/Mean_grad/Shape_1Const*
valueB:*
dtype0*<
_class2
0.loc:@gradients/discriminator_1/Mean_grad/Shape*
_output_shapes
:
?
/gradients/discriminator_1/Mean_grad/range/startConst*
value	B : *
dtype0*<
_class2
0.loc:@gradients/discriminator_1/Mean_grad/Shape*
_output_shapes
: 
?
/gradients/discriminator_1/Mean_grad/range/deltaConst*
value	B :*
dtype0*<
_class2
0.loc:@gradients/discriminator_1/Mean_grad/Shape*
_output_shapes
: 
?
)gradients/discriminator_1/Mean_grad/rangeRange/gradients/discriminator_1/Mean_grad/range/start(gradients/discriminator_1/Mean_grad/Size/gradients/discriminator_1/Mean_grad/range/delta*

Tidx0*<
_class2
0.loc:@gradients/discriminator_1/Mean_grad/Shape*
_output_shapes
:
?
.gradients/discriminator_1/Mean_grad/Fill/valueConst*
value	B :*
dtype0*<
_class2
0.loc:@gradients/discriminator_1/Mean_grad/Shape*
_output_shapes
: 
?
(gradients/discriminator_1/Mean_grad/FillFill+gradients/discriminator_1/Mean_grad/Shape_1.gradients/discriminator_1/Mean_grad/Fill/value*
T0*

index_type0*<
_class2
0.loc:@gradients/discriminator_1/Mean_grad/Shape*
_output_shapes
:
?
1gradients/discriminator_1/Mean_grad/DynamicStitchDynamicStitch)gradients/discriminator_1/Mean_grad/range'gradients/discriminator_1/Mean_grad/mod)gradients/discriminator_1/Mean_grad/Shape(gradients/discriminator_1/Mean_grad/Fill*
T0*
N*<
_class2
0.loc:@gradients/discriminator_1/Mean_grad/Shape*
_output_shapes
:
?
-gradients/discriminator_1/Mean_grad/Maximum/yConst*
value	B :*
dtype0*<
_class2
0.loc:@gradients/discriminator_1/Mean_grad/Shape*
_output_shapes
: 
?
+gradients/discriminator_1/Mean_grad/MaximumMaximum1gradients/discriminator_1/Mean_grad/DynamicStitch-gradients/discriminator_1/Mean_grad/Maximum/y*
T0*<
_class2
0.loc:@gradients/discriminator_1/Mean_grad/Shape*
_output_shapes
:
?
,gradients/discriminator_1/Mean_grad/floordivFloorDiv)gradients/discriminator_1/Mean_grad/Shape+gradients/discriminator_1/Mean_grad/Maximum*
T0*<
_class2
0.loc:@gradients/discriminator_1/Mean_grad/Shape*
_output_shapes
:
?
+gradients/discriminator_1/Mean_grad/ReshapeReshapeDgradients/discriminator_1/dense/MatMul_grad/tuple/control_dependency1gradients/discriminator_1/Mean_grad/DynamicStitch*
T0*
Tshape0*J
_output_shapes8
6:4????????????????????????????????????
?
(gradients/discriminator_1/Mean_grad/TileTile+gradients/discriminator_1/Mean_grad/Reshape,gradients/discriminator_1/Mean_grad/floordiv*
T0*

Tmultiples0*J
_output_shapes8
6:4????????????????????????????????????
?
+gradients/discriminator_1/Mean_grad/Shape_2Shapediscriminator_1/conv2d_5/Relu*
T0*
out_type0*
_output_shapes
:

+gradients/discriminator_1/Mean_grad/Shape_3Shapediscriminator_1/Mean*
T0*
out_type0*
_output_shapes
:
s
)gradients/discriminator_1/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
?
(gradients/discriminator_1/Mean_grad/ProdProd+gradients/discriminator_1/Mean_grad/Shape_2)gradients/discriminator_1/Mean_grad/Const*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
u
+gradients/discriminator_1/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
?
*gradients/discriminator_1/Mean_grad/Prod_1Prod+gradients/discriminator_1/Mean_grad/Shape_3+gradients/discriminator_1/Mean_grad/Const_1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
q
/gradients/discriminator_1/Mean_grad/Maximum_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
?
-gradients/discriminator_1/Mean_grad/Maximum_1Maximum*gradients/discriminator_1/Mean_grad/Prod_1/gradients/discriminator_1/Mean_grad/Maximum_1/y*
T0*
_output_shapes
: 
?
.gradients/discriminator_1/Mean_grad/floordiv_1FloorDiv(gradients/discriminator_1/Mean_grad/Prod-gradients/discriminator_1/Mean_grad/Maximum_1*
T0*
_output_shapes
: 
?
(gradients/discriminator_1/Mean_grad/CastCast.gradients/discriminator_1/Mean_grad/floordiv_1*

DstT0*
Truncate( *

SrcT0*
_output_shapes
: 
?
+gradients/discriminator_1/Mean_grad/truedivRealDiv(gradients/discriminator_1/Mean_grad/Tile(gradients/discriminator_1/Mean_grad/Cast*
T0*0
_output_shapes
:??????????
?
'gradients/discriminator/Mean_grad/ShapeShapediscriminator/conv2d_5/Relu*
T0*
out_type0*
_output_shapes
:
?
&gradients/discriminator/Mean_grad/SizeConst*
value	B :*
dtype0*:
_class0
.,loc:@gradients/discriminator/Mean_grad/Shape*
_output_shapes
: 
?
%gradients/discriminator/Mean_grad/addAdd$discriminator/Mean/reduction_indices&gradients/discriminator/Mean_grad/Size*
T0*:
_class0
.,loc:@gradients/discriminator/Mean_grad/Shape*
_output_shapes
:
?
%gradients/discriminator/Mean_grad/modFloorMod%gradients/discriminator/Mean_grad/add&gradients/discriminator/Mean_grad/Size*
T0*:
_class0
.,loc:@gradients/discriminator/Mean_grad/Shape*
_output_shapes
:
?
)gradients/discriminator/Mean_grad/Shape_1Const*
valueB:*
dtype0*:
_class0
.,loc:@gradients/discriminator/Mean_grad/Shape*
_output_shapes
:
?
-gradients/discriminator/Mean_grad/range/startConst*
value	B : *
dtype0*:
_class0
.,loc:@gradients/discriminator/Mean_grad/Shape*
_output_shapes
: 
?
-gradients/discriminator/Mean_grad/range/deltaConst*
value	B :*
dtype0*:
_class0
.,loc:@gradients/discriminator/Mean_grad/Shape*
_output_shapes
: 
?
'gradients/discriminator/Mean_grad/rangeRange-gradients/discriminator/Mean_grad/range/start&gradients/discriminator/Mean_grad/Size-gradients/discriminator/Mean_grad/range/delta*

Tidx0*:
_class0
.,loc:@gradients/discriminator/Mean_grad/Shape*
_output_shapes
:
?
,gradients/discriminator/Mean_grad/Fill/valueConst*
value	B :*
dtype0*:
_class0
.,loc:@gradients/discriminator/Mean_grad/Shape*
_output_shapes
: 
?
&gradients/discriminator/Mean_grad/FillFill)gradients/discriminator/Mean_grad/Shape_1,gradients/discriminator/Mean_grad/Fill/value*
T0*

index_type0*:
_class0
.,loc:@gradients/discriminator/Mean_grad/Shape*
_output_shapes
:
?
/gradients/discriminator/Mean_grad/DynamicStitchDynamicStitch'gradients/discriminator/Mean_grad/range%gradients/discriminator/Mean_grad/mod'gradients/discriminator/Mean_grad/Shape&gradients/discriminator/Mean_grad/Fill*
T0*
N*:
_class0
.,loc:@gradients/discriminator/Mean_grad/Shape*
_output_shapes
:
?
+gradients/discriminator/Mean_grad/Maximum/yConst*
value	B :*
dtype0*:
_class0
.,loc:@gradients/discriminator/Mean_grad/Shape*
_output_shapes
: 
?
)gradients/discriminator/Mean_grad/MaximumMaximum/gradients/discriminator/Mean_grad/DynamicStitch+gradients/discriminator/Mean_grad/Maximum/y*
T0*:
_class0
.,loc:@gradients/discriminator/Mean_grad/Shape*
_output_shapes
:
?
*gradients/discriminator/Mean_grad/floordivFloorDiv'gradients/discriminator/Mean_grad/Shape)gradients/discriminator/Mean_grad/Maximum*
T0*:
_class0
.,loc:@gradients/discriminator/Mean_grad/Shape*
_output_shapes
:
?
)gradients/discriminator/Mean_grad/ReshapeReshapeBgradients/discriminator/dense/MatMul_grad/tuple/control_dependency/gradients/discriminator/Mean_grad/DynamicStitch*
T0*
Tshape0*J
_output_shapes8
6:4????????????????????????????????????
?
&gradients/discriminator/Mean_grad/TileTile)gradients/discriminator/Mean_grad/Reshape*gradients/discriminator/Mean_grad/floordiv*
T0*

Tmultiples0*J
_output_shapes8
6:4????????????????????????????????????
?
)gradients/discriminator/Mean_grad/Shape_2Shapediscriminator/conv2d_5/Relu*
T0*
out_type0*
_output_shapes
:
{
)gradients/discriminator/Mean_grad/Shape_3Shapediscriminator/Mean*
T0*
out_type0*
_output_shapes
:
q
'gradients/discriminator/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
?
&gradients/discriminator/Mean_grad/ProdProd)gradients/discriminator/Mean_grad/Shape_2'gradients/discriminator/Mean_grad/Const*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
s
)gradients/discriminator/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
?
(gradients/discriminator/Mean_grad/Prod_1Prod)gradients/discriminator/Mean_grad/Shape_3)gradients/discriminator/Mean_grad/Const_1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
o
-gradients/discriminator/Mean_grad/Maximum_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
?
+gradients/discriminator/Mean_grad/Maximum_1Maximum(gradients/discriminator/Mean_grad/Prod_1-gradients/discriminator/Mean_grad/Maximum_1/y*
T0*
_output_shapes
: 
?
,gradients/discriminator/Mean_grad/floordiv_1FloorDiv&gradients/discriminator/Mean_grad/Prod+gradients/discriminator/Mean_grad/Maximum_1*
T0*
_output_shapes
: 
?
&gradients/discriminator/Mean_grad/CastCast,gradients/discriminator/Mean_grad/floordiv_1*

DstT0*
Truncate( *

SrcT0*
_output_shapes
: 
?
)gradients/discriminator/Mean_grad/truedivRealDiv&gradients/discriminator/Mean_grad/Tile&gradients/discriminator/Mean_grad/Cast*
T0*0
_output_shapes
:??????????
?
gradients/AddN_7AddNFgradients/discriminator_1/dense/MatMul_grad/tuple/control_dependency_1Dgradients/discriminator/dense/MatMul_grad/tuple/control_dependency_1*
T0*
N*G
_class=
;9loc:@gradients/discriminator_1/dense/MatMul_grad/MatMul_1*
_output_shapes
:	??
?
5gradients/discriminator_1/conv2d_5/Relu_grad/ReluGradReluGrad+gradients/discriminator_1/Mean_grad/truedivdiscriminator_1/conv2d_5/Relu*
T0*0
_output_shapes
:??????????
?
3gradients/discriminator/conv2d_5/Relu_grad/ReluGradReluGrad)gradients/discriminator/Mean_grad/truedivdiscriminator/conv2d_5/Relu*
T0*0
_output_shapes
:??????????
?
;gradients/discriminator_1/conv2d_5/BiasAdd_grad/BiasAddGradBiasAddGrad5gradients/discriminator_1/conv2d_5/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:?
?
@gradients/discriminator_1/conv2d_5/BiasAdd_grad/tuple/group_depsNoOp<^gradients/discriminator_1/conv2d_5/BiasAdd_grad/BiasAddGrad6^gradients/discriminator_1/conv2d_5/Relu_grad/ReluGrad
?
Hgradients/discriminator_1/conv2d_5/BiasAdd_grad/tuple/control_dependencyIdentity5gradients/discriminator_1/conv2d_5/Relu_grad/ReluGradA^gradients/discriminator_1/conv2d_5/BiasAdd_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/discriminator_1/conv2d_5/Relu_grad/ReluGrad*0
_output_shapes
:??????????
?
Jgradients/discriminator_1/conv2d_5/BiasAdd_grad/tuple/control_dependency_1Identity;gradients/discriminator_1/conv2d_5/BiasAdd_grad/BiasAddGradA^gradients/discriminator_1/conv2d_5/BiasAdd_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients/discriminator_1/conv2d_5/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:?
?
9gradients/discriminator/conv2d_5/BiasAdd_grad/BiasAddGradBiasAddGrad3gradients/discriminator/conv2d_5/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:?
?
>gradients/discriminator/conv2d_5/BiasAdd_grad/tuple/group_depsNoOp:^gradients/discriminator/conv2d_5/BiasAdd_grad/BiasAddGrad4^gradients/discriminator/conv2d_5/Relu_grad/ReluGrad
?
Fgradients/discriminator/conv2d_5/BiasAdd_grad/tuple/control_dependencyIdentity3gradients/discriminator/conv2d_5/Relu_grad/ReluGrad?^gradients/discriminator/conv2d_5/BiasAdd_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/discriminator/conv2d_5/Relu_grad/ReluGrad*0
_output_shapes
:??????????
?
Hgradients/discriminator/conv2d_5/BiasAdd_grad/tuple/control_dependency_1Identity9gradients/discriminator/conv2d_5/BiasAdd_grad/BiasAddGrad?^gradients/discriminator/conv2d_5/BiasAdd_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/discriminator/conv2d_5/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:?
?
5gradients/discriminator_1/conv2d_5/Conv2D_grad/ShapeNShapeNdiscriminator_1/Relu_2"discriminator/conv2d_5/kernel/read*
T0*
out_type0*
N* 
_output_shapes
::
?
Bgradients/discriminator_1/conv2d_5/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput5gradients/discriminator_1/conv2d_5/Conv2D_grad/ShapeN"discriminator/conv2d_5/kernel/readHgradients/discriminator_1/conv2d_5/BiasAdd_grad/tuple/control_dependency*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingVALID*
use_cudnn_on_gpu(*0
_output_shapes
:??????????
?
Cgradients/discriminator_1/conv2d_5/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterdiscriminator_1/Relu_27gradients/discriminator_1/conv2d_5/Conv2D_grad/ShapeN:1Hgradients/discriminator_1/conv2d_5/BiasAdd_grad/tuple/control_dependency*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingVALID*
use_cudnn_on_gpu(*(
_output_shapes
:??
?
?gradients/discriminator_1/conv2d_5/Conv2D_grad/tuple/group_depsNoOpD^gradients/discriminator_1/conv2d_5/Conv2D_grad/Conv2DBackpropFilterC^gradients/discriminator_1/conv2d_5/Conv2D_grad/Conv2DBackpropInput
?
Ggradients/discriminator_1/conv2d_5/Conv2D_grad/tuple/control_dependencyIdentityBgradients/discriminator_1/conv2d_5/Conv2D_grad/Conv2DBackpropInput@^gradients/discriminator_1/conv2d_5/Conv2D_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/discriminator_1/conv2d_5/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:??????????
?
Igradients/discriminator_1/conv2d_5/Conv2D_grad/tuple/control_dependency_1IdentityCgradients/discriminator_1/conv2d_5/Conv2D_grad/Conv2DBackpropFilter@^gradients/discriminator_1/conv2d_5/Conv2D_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients/discriminator_1/conv2d_5/Conv2D_grad/Conv2DBackpropFilter*(
_output_shapes
:??
?
3gradients/discriminator/conv2d_5/Conv2D_grad/ShapeNShapeNdiscriminator/Relu_2"discriminator/conv2d_5/kernel/read*
T0*
out_type0*
N* 
_output_shapes
::
?
@gradients/discriminator/conv2d_5/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput3gradients/discriminator/conv2d_5/Conv2D_grad/ShapeN"discriminator/conv2d_5/kernel/readFgradients/discriminator/conv2d_5/BiasAdd_grad/tuple/control_dependency*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingVALID*
use_cudnn_on_gpu(*0
_output_shapes
:??????????
?
Agradients/discriminator/conv2d_5/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterdiscriminator/Relu_25gradients/discriminator/conv2d_5/Conv2D_grad/ShapeN:1Fgradients/discriminator/conv2d_5/BiasAdd_grad/tuple/control_dependency*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingVALID*
use_cudnn_on_gpu(*(
_output_shapes
:??
?
=gradients/discriminator/conv2d_5/Conv2D_grad/tuple/group_depsNoOpB^gradients/discriminator/conv2d_5/Conv2D_grad/Conv2DBackpropFilterA^gradients/discriminator/conv2d_5/Conv2D_grad/Conv2DBackpropInput
?
Egradients/discriminator/conv2d_5/Conv2D_grad/tuple/control_dependencyIdentity@gradients/discriminator/conv2d_5/Conv2D_grad/Conv2DBackpropInput>^gradients/discriminator/conv2d_5/Conv2D_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/discriminator/conv2d_5/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:??????????
?
Ggradients/discriminator/conv2d_5/Conv2D_grad/tuple/control_dependency_1IdentityAgradients/discriminator/conv2d_5/Conv2D_grad/Conv2DBackpropFilter>^gradients/discriminator/conv2d_5/Conv2D_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients/discriminator/conv2d_5/Conv2D_grad/Conv2DBackpropFilter*(
_output_shapes
:??
?
gradients/AddN_8AddNJgradients/discriminator_1/conv2d_5/BiasAdd_grad/tuple/control_dependency_1Hgradients/discriminator/conv2d_5/BiasAdd_grad/tuple/control_dependency_1*
T0*
N*N
_classD
B@loc:@gradients/discriminator_1/conv2d_5/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:?
?
.gradients/discriminator_1/Relu_2_grad/ReluGradReluGradGgradients/discriminator_1/conv2d_5/Conv2D_grad/tuple/control_dependencydiscriminator_1/Relu_2*
T0*0
_output_shapes
:??????????
?
,gradients/discriminator/Relu_2_grad/ReluGradReluGradEgradients/discriminator/conv2d_5/Conv2D_grad/tuple/control_dependencydiscriminator/Relu_2*
T0*0
_output_shapes
:??????????
?
gradients/AddN_9AddNIgradients/discriminator_1/conv2d_5/Conv2D_grad/tuple/control_dependency_1Ggradients/discriminator/conv2d_5/Conv2D_grad/tuple/control_dependency_1*
T0*
N*V
_classL
JHloc:@gradients/discriminator_1/conv2d_5/Conv2D_grad/Conv2DBackpropFilter*(
_output_shapes
:??
?
gradients/zeros_like_1	ZerosLike6discriminator_1/batch_normalization_2/FusedBatchNorm:1*
T0*
_output_shapes	
:?
?
gradients/zeros_like_2	ZerosLike6discriminator_1/batch_normalization_2/FusedBatchNorm:2*
T0*
_output_shapes	
:?
?
gradients/zeros_like_3	ZerosLike6discriminator_1/batch_normalization_2/FusedBatchNorm:3*
T0*
_output_shapes	
:?
?
gradients/zeros_like_4	ZerosLike6discriminator_1/batch_normalization_2/FusedBatchNorm:4*
T0*
_output_shapes	
:?
?
Vgradients/discriminator_1/batch_normalization_2/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGrad.gradients/discriminator_1/Relu_2_grad/ReluGrad discriminator_1/conv2d_4/BiasAdd.discriminator/batch_normalization_2/gamma/read4discriminator/batch_normalization_2/moving_mean/read8discriminator/batch_normalization_2/moving_variance/read*
is_training( *
T0*
data_formatNHWC*
epsilon%o?:*L
_output_shapes:
8:??????????:?:?:?:?
?
Tgradients/discriminator_1/batch_normalization_2/FusedBatchNorm_grad/tuple/group_depsNoOpW^gradients/discriminator_1/batch_normalization_2/FusedBatchNorm_grad/FusedBatchNormGrad
?
\gradients/discriminator_1/batch_normalization_2/FusedBatchNorm_grad/tuple/control_dependencyIdentityVgradients/discriminator_1/batch_normalization_2/FusedBatchNorm_grad/FusedBatchNormGradU^gradients/discriminator_1/batch_normalization_2/FusedBatchNorm_grad/tuple/group_deps*
T0*i
_class_
][loc:@gradients/discriminator_1/batch_normalization_2/FusedBatchNorm_grad/FusedBatchNormGrad*0
_output_shapes
:??????????
?
^gradients/discriminator_1/batch_normalization_2/FusedBatchNorm_grad/tuple/control_dependency_1IdentityXgradients/discriminator_1/batch_normalization_2/FusedBatchNorm_grad/FusedBatchNormGrad:1U^gradients/discriminator_1/batch_normalization_2/FusedBatchNorm_grad/tuple/group_deps*
T0*i
_class_
][loc:@gradients/discriminator_1/batch_normalization_2/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes	
:?
?
^gradients/discriminator_1/batch_normalization_2/FusedBatchNorm_grad/tuple/control_dependency_2IdentityXgradients/discriminator_1/batch_normalization_2/FusedBatchNorm_grad/FusedBatchNormGrad:2U^gradients/discriminator_1/batch_normalization_2/FusedBatchNorm_grad/tuple/group_deps*
T0*i
_class_
][loc:@gradients/discriminator_1/batch_normalization_2/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes	
:?

gradients/zeros_like_5	ZerosLike4discriminator/batch_normalization_2/FusedBatchNorm:1*
T0*
_output_shapes	
:?

gradients/zeros_like_6	ZerosLike4discriminator/batch_normalization_2/FusedBatchNorm:2*
T0*
_output_shapes	
:?

gradients/zeros_like_7	ZerosLike4discriminator/batch_normalization_2/FusedBatchNorm:3*
T0*
_output_shapes	
:?

gradients/zeros_like_8	ZerosLike4discriminator/batch_normalization_2/FusedBatchNorm:4*
T0*
_output_shapes	
:?
?
Tgradients/discriminator/batch_normalization_2/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGrad,gradients/discriminator/Relu_2_grad/ReluGraddiscriminator/conv2d_4/BiasAdd.discriminator/batch_normalization_2/gamma/read4discriminator/batch_normalization_2/moving_mean/read8discriminator/batch_normalization_2/moving_variance/read*
is_training( *
T0*
data_formatNHWC*
epsilon%o?:*L
_output_shapes:
8:??????????:?:?:?:?
?
Rgradients/discriminator/batch_normalization_2/FusedBatchNorm_grad/tuple/group_depsNoOpU^gradients/discriminator/batch_normalization_2/FusedBatchNorm_grad/FusedBatchNormGrad
?
Zgradients/discriminator/batch_normalization_2/FusedBatchNorm_grad/tuple/control_dependencyIdentityTgradients/discriminator/batch_normalization_2/FusedBatchNorm_grad/FusedBatchNormGradS^gradients/discriminator/batch_normalization_2/FusedBatchNorm_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients/discriminator/batch_normalization_2/FusedBatchNorm_grad/FusedBatchNormGrad*0
_output_shapes
:??????????
?
\gradients/discriminator/batch_normalization_2/FusedBatchNorm_grad/tuple/control_dependency_1IdentityVgradients/discriminator/batch_normalization_2/FusedBatchNorm_grad/FusedBatchNormGrad:1S^gradients/discriminator/batch_normalization_2/FusedBatchNorm_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients/discriminator/batch_normalization_2/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes	
:?
?
\gradients/discriminator/batch_normalization_2/FusedBatchNorm_grad/tuple/control_dependency_2IdentityVgradients/discriminator/batch_normalization_2/FusedBatchNorm_grad/FusedBatchNormGrad:2S^gradients/discriminator/batch_normalization_2/FusedBatchNorm_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients/discriminator/batch_normalization_2/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes	
:?
?
;gradients/discriminator_1/conv2d_4/BiasAdd_grad/BiasAddGradBiasAddGrad\gradients/discriminator_1/batch_normalization_2/FusedBatchNorm_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes	
:?
?
@gradients/discriminator_1/conv2d_4/BiasAdd_grad/tuple/group_depsNoOp]^gradients/discriminator_1/batch_normalization_2/FusedBatchNorm_grad/tuple/control_dependency<^gradients/discriminator_1/conv2d_4/BiasAdd_grad/BiasAddGrad
?
Hgradients/discriminator_1/conv2d_4/BiasAdd_grad/tuple/control_dependencyIdentity\gradients/discriminator_1/batch_normalization_2/FusedBatchNorm_grad/tuple/control_dependencyA^gradients/discriminator_1/conv2d_4/BiasAdd_grad/tuple/group_deps*
T0*i
_class_
][loc:@gradients/discriminator_1/batch_normalization_2/FusedBatchNorm_grad/FusedBatchNormGrad*0
_output_shapes
:??????????
?
Jgradients/discriminator_1/conv2d_4/BiasAdd_grad/tuple/control_dependency_1Identity;gradients/discriminator_1/conv2d_4/BiasAdd_grad/BiasAddGradA^gradients/discriminator_1/conv2d_4/BiasAdd_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients/discriminator_1/conv2d_4/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:?
?
9gradients/discriminator/conv2d_4/BiasAdd_grad/BiasAddGradBiasAddGradZgradients/discriminator/batch_normalization_2/FusedBatchNorm_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes	
:?
?
>gradients/discriminator/conv2d_4/BiasAdd_grad/tuple/group_depsNoOp[^gradients/discriminator/batch_normalization_2/FusedBatchNorm_grad/tuple/control_dependency:^gradients/discriminator/conv2d_4/BiasAdd_grad/BiasAddGrad
?
Fgradients/discriminator/conv2d_4/BiasAdd_grad/tuple/control_dependencyIdentityZgradients/discriminator/batch_normalization_2/FusedBatchNorm_grad/tuple/control_dependency?^gradients/discriminator/conv2d_4/BiasAdd_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients/discriminator/batch_normalization_2/FusedBatchNorm_grad/FusedBatchNormGrad*0
_output_shapes
:??????????
?
Hgradients/discriminator/conv2d_4/BiasAdd_grad/tuple/control_dependency_1Identity9gradients/discriminator/conv2d_4/BiasAdd_grad/BiasAddGrad?^gradients/discriminator/conv2d_4/BiasAdd_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/discriminator/conv2d_4/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:?
?
5gradients/discriminator_1/conv2d_4/Conv2D_grad/ShapeNShapeNdiscriminator_1/Relu_1"discriminator/conv2d_4/kernel/read*
T0*
out_type0*
N* 
_output_shapes
::
?
Bgradients/discriminator_1/conv2d_4/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput5gradients/discriminator_1/conv2d_4/Conv2D_grad/ShapeN"discriminator/conv2d_4/kernel/readHgradients/discriminator_1/conv2d_4/BiasAdd_grad/tuple/control_dependency*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(*0
_output_shapes
:??????????
?
Cgradients/discriminator_1/conv2d_4/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterdiscriminator_1/Relu_17gradients/discriminator_1/conv2d_4/Conv2D_grad/ShapeN:1Hgradients/discriminator_1/conv2d_4/BiasAdd_grad/tuple/control_dependency*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(*(
_output_shapes
:??
?
?gradients/discriminator_1/conv2d_4/Conv2D_grad/tuple/group_depsNoOpD^gradients/discriminator_1/conv2d_4/Conv2D_grad/Conv2DBackpropFilterC^gradients/discriminator_1/conv2d_4/Conv2D_grad/Conv2DBackpropInput
?
Ggradients/discriminator_1/conv2d_4/Conv2D_grad/tuple/control_dependencyIdentityBgradients/discriminator_1/conv2d_4/Conv2D_grad/Conv2DBackpropInput@^gradients/discriminator_1/conv2d_4/Conv2D_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/discriminator_1/conv2d_4/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:??????????
?
Igradients/discriminator_1/conv2d_4/Conv2D_grad/tuple/control_dependency_1IdentityCgradients/discriminator_1/conv2d_4/Conv2D_grad/Conv2DBackpropFilter@^gradients/discriminator_1/conv2d_4/Conv2D_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients/discriminator_1/conv2d_4/Conv2D_grad/Conv2DBackpropFilter*(
_output_shapes
:??
?
3gradients/discriminator/conv2d_4/Conv2D_grad/ShapeNShapeNdiscriminator/Relu_1"discriminator/conv2d_4/kernel/read*
T0*
out_type0*
N* 
_output_shapes
::
?
@gradients/discriminator/conv2d_4/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput3gradients/discriminator/conv2d_4/Conv2D_grad/ShapeN"discriminator/conv2d_4/kernel/readFgradients/discriminator/conv2d_4/BiasAdd_grad/tuple/control_dependency*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(*0
_output_shapes
:??????????
?
Agradients/discriminator/conv2d_4/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterdiscriminator/Relu_15gradients/discriminator/conv2d_4/Conv2D_grad/ShapeN:1Fgradients/discriminator/conv2d_4/BiasAdd_grad/tuple/control_dependency*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(*(
_output_shapes
:??
?
=gradients/discriminator/conv2d_4/Conv2D_grad/tuple/group_depsNoOpB^gradients/discriminator/conv2d_4/Conv2D_grad/Conv2DBackpropFilterA^gradients/discriminator/conv2d_4/Conv2D_grad/Conv2DBackpropInput
?
Egradients/discriminator/conv2d_4/Conv2D_grad/tuple/control_dependencyIdentity@gradients/discriminator/conv2d_4/Conv2D_grad/Conv2DBackpropInput>^gradients/discriminator/conv2d_4/Conv2D_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/discriminator/conv2d_4/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:??????????
?
Ggradients/discriminator/conv2d_4/Conv2D_grad/tuple/control_dependency_1IdentityAgradients/discriminator/conv2d_4/Conv2D_grad/Conv2DBackpropFilter>^gradients/discriminator/conv2d_4/Conv2D_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients/discriminator/conv2d_4/Conv2D_grad/Conv2DBackpropFilter*(
_output_shapes
:??
?
gradients/AddN_10AddNJgradients/discriminator_1/conv2d_4/BiasAdd_grad/tuple/control_dependency_1Hgradients/discriminator/conv2d_4/BiasAdd_grad/tuple/control_dependency_1*
T0*
N*N
_classD
B@loc:@gradients/discriminator_1/conv2d_4/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:?
?
.gradients/discriminator_1/Relu_1_grad/ReluGradReluGradGgradients/discriminator_1/conv2d_4/Conv2D_grad/tuple/control_dependencydiscriminator_1/Relu_1*
T0*0
_output_shapes
:??????????
?
,gradients/discriminator/Relu_1_grad/ReluGradReluGradEgradients/discriminator/conv2d_4/Conv2D_grad/tuple/control_dependencydiscriminator/Relu_1*
T0*0
_output_shapes
:??????????
?
gradients/AddN_11AddNIgradients/discriminator_1/conv2d_4/Conv2D_grad/tuple/control_dependency_1Ggradients/discriminator/conv2d_4/Conv2D_grad/tuple/control_dependency_1*
T0*
N*V
_classL
JHloc:@gradients/discriminator_1/conv2d_4/Conv2D_grad/Conv2DBackpropFilter*(
_output_shapes
:??
?
gradients/zeros_like_9	ZerosLike6discriminator_1/batch_normalization_1/FusedBatchNorm:1*
T0*
_output_shapes	
:?
?
gradients/zeros_like_10	ZerosLike6discriminator_1/batch_normalization_1/FusedBatchNorm:2*
T0*
_output_shapes	
:?
?
gradients/zeros_like_11	ZerosLike6discriminator_1/batch_normalization_1/FusedBatchNorm:3*
T0*
_output_shapes	
:?
?
gradients/zeros_like_12	ZerosLike6discriminator_1/batch_normalization_1/FusedBatchNorm:4*
T0*
_output_shapes	
:?
?
Vgradients/discriminator_1/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGrad.gradients/discriminator_1/Relu_1_grad/ReluGrad discriminator_1/conv2d_3/BiasAdd.discriminator/batch_normalization_1/gamma/read4discriminator/batch_normalization_1/moving_mean/read8discriminator/batch_normalization_1/moving_variance/read*
is_training( *
T0*
data_formatNHWC*
epsilon%o?:*L
_output_shapes:
8:??????????:?:?:?:?
?
Tgradients/discriminator_1/batch_normalization_1/FusedBatchNorm_grad/tuple/group_depsNoOpW^gradients/discriminator_1/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGrad
?
\gradients/discriminator_1/batch_normalization_1/FusedBatchNorm_grad/tuple/control_dependencyIdentityVgradients/discriminator_1/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGradU^gradients/discriminator_1/batch_normalization_1/FusedBatchNorm_grad/tuple/group_deps*
T0*i
_class_
][loc:@gradients/discriminator_1/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGrad*0
_output_shapes
:??????????
?
^gradients/discriminator_1/batch_normalization_1/FusedBatchNorm_grad/tuple/control_dependency_1IdentityXgradients/discriminator_1/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGrad:1U^gradients/discriminator_1/batch_normalization_1/FusedBatchNorm_grad/tuple/group_deps*
T0*i
_class_
][loc:@gradients/discriminator_1/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes	
:?
?
^gradients/discriminator_1/batch_normalization_1/FusedBatchNorm_grad/tuple/control_dependency_2IdentityXgradients/discriminator_1/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGrad:2U^gradients/discriminator_1/batch_normalization_1/FusedBatchNorm_grad/tuple/group_deps*
T0*i
_class_
][loc:@gradients/discriminator_1/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes	
:?
?
gradients/zeros_like_13	ZerosLike4discriminator/batch_normalization_1/FusedBatchNorm:1*
T0*
_output_shapes	
:?
?
gradients/zeros_like_14	ZerosLike4discriminator/batch_normalization_1/FusedBatchNorm:2*
T0*
_output_shapes	
:?
?
gradients/zeros_like_15	ZerosLike4discriminator/batch_normalization_1/FusedBatchNorm:3*
T0*
_output_shapes	
:?
?
gradients/zeros_like_16	ZerosLike4discriminator/batch_normalization_1/FusedBatchNorm:4*
T0*
_output_shapes	
:?
?
Tgradients/discriminator/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGrad,gradients/discriminator/Relu_1_grad/ReluGraddiscriminator/conv2d_3/BiasAdd.discriminator/batch_normalization_1/gamma/read4discriminator/batch_normalization_1/moving_mean/read8discriminator/batch_normalization_1/moving_variance/read*
is_training( *
T0*
data_formatNHWC*
epsilon%o?:*L
_output_shapes:
8:??????????:?:?:?:?
?
Rgradients/discriminator/batch_normalization_1/FusedBatchNorm_grad/tuple/group_depsNoOpU^gradients/discriminator/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGrad
?
Zgradients/discriminator/batch_normalization_1/FusedBatchNorm_grad/tuple/control_dependencyIdentityTgradients/discriminator/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGradS^gradients/discriminator/batch_normalization_1/FusedBatchNorm_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients/discriminator/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGrad*0
_output_shapes
:??????????
?
\gradients/discriminator/batch_normalization_1/FusedBatchNorm_grad/tuple/control_dependency_1IdentityVgradients/discriminator/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGrad:1S^gradients/discriminator/batch_normalization_1/FusedBatchNorm_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients/discriminator/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes	
:?
?
\gradients/discriminator/batch_normalization_1/FusedBatchNorm_grad/tuple/control_dependency_2IdentityVgradients/discriminator/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGrad:2S^gradients/discriminator/batch_normalization_1/FusedBatchNorm_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients/discriminator/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes	
:?
?
;gradients/discriminator_1/conv2d_3/BiasAdd_grad/BiasAddGradBiasAddGrad\gradients/discriminator_1/batch_normalization_1/FusedBatchNorm_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes	
:?
?
@gradients/discriminator_1/conv2d_3/BiasAdd_grad/tuple/group_depsNoOp]^gradients/discriminator_1/batch_normalization_1/FusedBatchNorm_grad/tuple/control_dependency<^gradients/discriminator_1/conv2d_3/BiasAdd_grad/BiasAddGrad
?
Hgradients/discriminator_1/conv2d_3/BiasAdd_grad/tuple/control_dependencyIdentity\gradients/discriminator_1/batch_normalization_1/FusedBatchNorm_grad/tuple/control_dependencyA^gradients/discriminator_1/conv2d_3/BiasAdd_grad/tuple/group_deps*
T0*i
_class_
][loc:@gradients/discriminator_1/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGrad*0
_output_shapes
:??????????
?
Jgradients/discriminator_1/conv2d_3/BiasAdd_grad/tuple/control_dependency_1Identity;gradients/discriminator_1/conv2d_3/BiasAdd_grad/BiasAddGradA^gradients/discriminator_1/conv2d_3/BiasAdd_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients/discriminator_1/conv2d_3/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:?
?
9gradients/discriminator/conv2d_3/BiasAdd_grad/BiasAddGradBiasAddGradZgradients/discriminator/batch_normalization_1/FusedBatchNorm_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes	
:?
?
>gradients/discriminator/conv2d_3/BiasAdd_grad/tuple/group_depsNoOp[^gradients/discriminator/batch_normalization_1/FusedBatchNorm_grad/tuple/control_dependency:^gradients/discriminator/conv2d_3/BiasAdd_grad/BiasAddGrad
?
Fgradients/discriminator/conv2d_3/BiasAdd_grad/tuple/control_dependencyIdentityZgradients/discriminator/batch_normalization_1/FusedBatchNorm_grad/tuple/control_dependency?^gradients/discriminator/conv2d_3/BiasAdd_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients/discriminator/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGrad*0
_output_shapes
:??????????
?
Hgradients/discriminator/conv2d_3/BiasAdd_grad/tuple/control_dependency_1Identity9gradients/discriminator/conv2d_3/BiasAdd_grad/BiasAddGrad?^gradients/discriminator/conv2d_3/BiasAdd_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/discriminator/conv2d_3/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:?
?
5gradients/discriminator_1/conv2d_3/Conv2D_grad/ShapeNShapeN"discriminator_1/dropout_2/Identity"discriminator/conv2d_3/kernel/read*
T0*
out_type0*
N* 
_output_shapes
::
?
Bgradients/discriminator_1/conv2d_3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput5gradients/discriminator_1/conv2d_3/Conv2D_grad/ShapeN"discriminator/conv2d_3/kernel/readHgradients/discriminator_1/conv2d_3/BiasAdd_grad/tuple/control_dependency*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(*/
_output_shapes
:?????????@
?
Cgradients/discriminator_1/conv2d_3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter"discriminator_1/dropout_2/Identity7gradients/discriminator_1/conv2d_3/Conv2D_grad/ShapeN:1Hgradients/discriminator_1/conv2d_3/BiasAdd_grad/tuple/control_dependency*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(*'
_output_shapes
:@?
?
?gradients/discriminator_1/conv2d_3/Conv2D_grad/tuple/group_depsNoOpD^gradients/discriminator_1/conv2d_3/Conv2D_grad/Conv2DBackpropFilterC^gradients/discriminator_1/conv2d_3/Conv2D_grad/Conv2DBackpropInput
?
Ggradients/discriminator_1/conv2d_3/Conv2D_grad/tuple/control_dependencyIdentityBgradients/discriminator_1/conv2d_3/Conv2D_grad/Conv2DBackpropInput@^gradients/discriminator_1/conv2d_3/Conv2D_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/discriminator_1/conv2d_3/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:?????????@
?
Igradients/discriminator_1/conv2d_3/Conv2D_grad/tuple/control_dependency_1IdentityCgradients/discriminator_1/conv2d_3/Conv2D_grad/Conv2DBackpropFilter@^gradients/discriminator_1/conv2d_3/Conv2D_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients/discriminator_1/conv2d_3/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:@?
?
3gradients/discriminator/conv2d_3/Conv2D_grad/ShapeNShapeN discriminator/dropout_2/Identity"discriminator/conv2d_3/kernel/read*
T0*
out_type0*
N* 
_output_shapes
::
?
@gradients/discriminator/conv2d_3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput3gradients/discriminator/conv2d_3/Conv2D_grad/ShapeN"discriminator/conv2d_3/kernel/readFgradients/discriminator/conv2d_3/BiasAdd_grad/tuple/control_dependency*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(*/
_output_shapes
:?????????@
?
Agradients/discriminator/conv2d_3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter discriminator/dropout_2/Identity5gradients/discriminator/conv2d_3/Conv2D_grad/ShapeN:1Fgradients/discriminator/conv2d_3/BiasAdd_grad/tuple/control_dependency*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(*'
_output_shapes
:@?
?
=gradients/discriminator/conv2d_3/Conv2D_grad/tuple/group_depsNoOpB^gradients/discriminator/conv2d_3/Conv2D_grad/Conv2DBackpropFilterA^gradients/discriminator/conv2d_3/Conv2D_grad/Conv2DBackpropInput
?
Egradients/discriminator/conv2d_3/Conv2D_grad/tuple/control_dependencyIdentity@gradients/discriminator/conv2d_3/Conv2D_grad/Conv2DBackpropInput>^gradients/discriminator/conv2d_3/Conv2D_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/discriminator/conv2d_3/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:?????????@
?
Ggradients/discriminator/conv2d_3/Conv2D_grad/tuple/control_dependency_1IdentityAgradients/discriminator/conv2d_3/Conv2D_grad/Conv2DBackpropFilter>^gradients/discriminator/conv2d_3/Conv2D_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients/discriminator/conv2d_3/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:@?
?
gradients/AddN_12AddNJgradients/discriminator_1/conv2d_3/BiasAdd_grad/tuple/control_dependency_1Hgradients/discriminator/conv2d_3/BiasAdd_grad/tuple/control_dependency_1*
T0*
N*N
_classD
B@loc:@gradients/discriminator_1/conv2d_3/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:?
?
gradients/AddN_13AddNIgradients/discriminator_1/conv2d_3/Conv2D_grad/tuple/control_dependency_1Ggradients/discriminator/conv2d_3/Conv2D_grad/tuple/control_dependency_1*
T0*
N*V
_classL
JHloc:@gradients/discriminator_1/conv2d_3/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:@?
?
,gradients/discriminator_1/Relu_grad/ReluGradReluGradGgradients/discriminator_1/conv2d_3/Conv2D_grad/tuple/control_dependencydiscriminator_1/Relu*
T0*/
_output_shapes
:?????????@
?
*gradients/discriminator/Relu_grad/ReluGradReluGradEgradients/discriminator/conv2d_3/Conv2D_grad/tuple/control_dependencydiscriminator/Relu*
T0*/
_output_shapes
:?????????@

gradients/zeros_like_17	ZerosLike4discriminator_1/batch_normalization/FusedBatchNorm:1*
T0*
_output_shapes
:@

gradients/zeros_like_18	ZerosLike4discriminator_1/batch_normalization/FusedBatchNorm:2*
T0*
_output_shapes
:@

gradients/zeros_like_19	ZerosLike4discriminator_1/batch_normalization/FusedBatchNorm:3*
T0*
_output_shapes
:@

gradients/zeros_like_20	ZerosLike4discriminator_1/batch_normalization/FusedBatchNorm:4*
T0*
_output_shapes
:@
?
Tgradients/discriminator_1/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGrad,gradients/discriminator_1/Relu_grad/ReluGrad discriminator_1/conv2d_2/BiasAdd,discriminator/batch_normalization/gamma/read2discriminator/batch_normalization/moving_mean/read6discriminator/batch_normalization/moving_variance/read*
is_training( *
T0*
data_formatNHWC*
epsilon%o?:*G
_output_shapes5
3:?????????@:@:@:@:@
?
Rgradients/discriminator_1/batch_normalization/FusedBatchNorm_grad/tuple/group_depsNoOpU^gradients/discriminator_1/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGrad
?
Zgradients/discriminator_1/batch_normalization/FusedBatchNorm_grad/tuple/control_dependencyIdentityTgradients/discriminator_1/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGradS^gradients/discriminator_1/batch_normalization/FusedBatchNorm_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients/discriminator_1/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGrad*/
_output_shapes
:?????????@
?
\gradients/discriminator_1/batch_normalization/FusedBatchNorm_grad/tuple/control_dependency_1IdentityVgradients/discriminator_1/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGrad:1S^gradients/discriminator_1/batch_normalization/FusedBatchNorm_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients/discriminator_1/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
:@
?
\gradients/discriminator_1/batch_normalization/FusedBatchNorm_grad/tuple/control_dependency_2IdentityVgradients/discriminator_1/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGrad:2S^gradients/discriminator_1/batch_normalization/FusedBatchNorm_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients/discriminator_1/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
:@
}
gradients/zeros_like_21	ZerosLike2discriminator/batch_normalization/FusedBatchNorm:1*
T0*
_output_shapes
:@
}
gradients/zeros_like_22	ZerosLike2discriminator/batch_normalization/FusedBatchNorm:2*
T0*
_output_shapes
:@
}
gradients/zeros_like_23	ZerosLike2discriminator/batch_normalization/FusedBatchNorm:3*
T0*
_output_shapes
:@
}
gradients/zeros_like_24	ZerosLike2discriminator/batch_normalization/FusedBatchNorm:4*
T0*
_output_shapes
:@
?
Rgradients/discriminator/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGrad*gradients/discriminator/Relu_grad/ReluGraddiscriminator/conv2d_2/BiasAdd,discriminator/batch_normalization/gamma/read2discriminator/batch_normalization/moving_mean/read6discriminator/batch_normalization/moving_variance/read*
is_training( *
T0*
data_formatNHWC*
epsilon%o?:*G
_output_shapes5
3:?????????@:@:@:@:@
?
Pgradients/discriminator/batch_normalization/FusedBatchNorm_grad/tuple/group_depsNoOpS^gradients/discriminator/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGrad
?
Xgradients/discriminator/batch_normalization/FusedBatchNorm_grad/tuple/control_dependencyIdentityRgradients/discriminator/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGradQ^gradients/discriminator/batch_normalization/FusedBatchNorm_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients/discriminator/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGrad*/
_output_shapes
:?????????@
?
Zgradients/discriminator/batch_normalization/FusedBatchNorm_grad/tuple/control_dependency_1IdentityTgradients/discriminator/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGrad:1Q^gradients/discriminator/batch_normalization/FusedBatchNorm_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients/discriminator/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
:@
?
Zgradients/discriminator/batch_normalization/FusedBatchNorm_grad/tuple/control_dependency_2IdentityTgradients/discriminator/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGrad:2Q^gradients/discriminator/batch_normalization/FusedBatchNorm_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients/discriminator/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
:@
?
;gradients/discriminator_1/conv2d_2/BiasAdd_grad/BiasAddGradBiasAddGradZgradients/discriminator_1/batch_normalization/FusedBatchNorm_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes
:@
?
@gradients/discriminator_1/conv2d_2/BiasAdd_grad/tuple/group_depsNoOp[^gradients/discriminator_1/batch_normalization/FusedBatchNorm_grad/tuple/control_dependency<^gradients/discriminator_1/conv2d_2/BiasAdd_grad/BiasAddGrad
?
Hgradients/discriminator_1/conv2d_2/BiasAdd_grad/tuple/control_dependencyIdentityZgradients/discriminator_1/batch_normalization/FusedBatchNorm_grad/tuple/control_dependencyA^gradients/discriminator_1/conv2d_2/BiasAdd_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients/discriminator_1/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGrad*/
_output_shapes
:?????????@
?
Jgradients/discriminator_1/conv2d_2/BiasAdd_grad/tuple/control_dependency_1Identity;gradients/discriminator_1/conv2d_2/BiasAdd_grad/BiasAddGradA^gradients/discriminator_1/conv2d_2/BiasAdd_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients/discriminator_1/conv2d_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
?
9gradients/discriminator/conv2d_2/BiasAdd_grad/BiasAddGradBiasAddGradXgradients/discriminator/batch_normalization/FusedBatchNorm_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes
:@
?
>gradients/discriminator/conv2d_2/BiasAdd_grad/tuple/group_depsNoOpY^gradients/discriminator/batch_normalization/FusedBatchNorm_grad/tuple/control_dependency:^gradients/discriminator/conv2d_2/BiasAdd_grad/BiasAddGrad
?
Fgradients/discriminator/conv2d_2/BiasAdd_grad/tuple/control_dependencyIdentityXgradients/discriminator/batch_normalization/FusedBatchNorm_grad/tuple/control_dependency?^gradients/discriminator/conv2d_2/BiasAdd_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients/discriminator/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGrad*/
_output_shapes
:?????????@
?
Hgradients/discriminator/conv2d_2/BiasAdd_grad/tuple/control_dependency_1Identity9gradients/discriminator/conv2d_2/BiasAdd_grad/BiasAddGrad?^gradients/discriminator/conv2d_2/BiasAdd_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/discriminator/conv2d_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
?
5gradients/discriminator_1/conv2d_2/Conv2D_grad/ShapeNShapeNdiscriminator_1/conv2d_1/Relu"discriminator/conv2d_2/kernel/read*
T0*
out_type0*
N* 
_output_shapes
::
?
Bgradients/discriminator_1/conv2d_2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput5gradients/discriminator_1/conv2d_2/Conv2D_grad/ShapeN"discriminator/conv2d_2/kernel/readHgradients/discriminator_1/conv2d_2/BiasAdd_grad/tuple/control_dependency*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(*/
_output_shapes
:?????????@
?
Cgradients/discriminator_1/conv2d_2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterdiscriminator_1/conv2d_1/Relu7gradients/discriminator_1/conv2d_2/Conv2D_grad/ShapeN:1Hgradients/discriminator_1/conv2d_2/BiasAdd_grad/tuple/control_dependency*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(*&
_output_shapes
:@@
?
?gradients/discriminator_1/conv2d_2/Conv2D_grad/tuple/group_depsNoOpD^gradients/discriminator_1/conv2d_2/Conv2D_grad/Conv2DBackpropFilterC^gradients/discriminator_1/conv2d_2/Conv2D_grad/Conv2DBackpropInput
?
Ggradients/discriminator_1/conv2d_2/Conv2D_grad/tuple/control_dependencyIdentityBgradients/discriminator_1/conv2d_2/Conv2D_grad/Conv2DBackpropInput@^gradients/discriminator_1/conv2d_2/Conv2D_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/discriminator_1/conv2d_2/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:?????????@
?
Igradients/discriminator_1/conv2d_2/Conv2D_grad/tuple/control_dependency_1IdentityCgradients/discriminator_1/conv2d_2/Conv2D_grad/Conv2DBackpropFilter@^gradients/discriminator_1/conv2d_2/Conv2D_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients/discriminator_1/conv2d_2/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:@@
?
3gradients/discriminator/conv2d_2/Conv2D_grad/ShapeNShapeNdiscriminator/conv2d_1/Relu"discriminator/conv2d_2/kernel/read*
T0*
out_type0*
N* 
_output_shapes
::
?
@gradients/discriminator/conv2d_2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput3gradients/discriminator/conv2d_2/Conv2D_grad/ShapeN"discriminator/conv2d_2/kernel/readFgradients/discriminator/conv2d_2/BiasAdd_grad/tuple/control_dependency*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(*/
_output_shapes
:?????????@
?
Agradients/discriminator/conv2d_2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterdiscriminator/conv2d_1/Relu5gradients/discriminator/conv2d_2/Conv2D_grad/ShapeN:1Fgradients/discriminator/conv2d_2/BiasAdd_grad/tuple/control_dependency*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(*&
_output_shapes
:@@
?
=gradients/discriminator/conv2d_2/Conv2D_grad/tuple/group_depsNoOpB^gradients/discriminator/conv2d_2/Conv2D_grad/Conv2DBackpropFilterA^gradients/discriminator/conv2d_2/Conv2D_grad/Conv2DBackpropInput
?
Egradients/discriminator/conv2d_2/Conv2D_grad/tuple/control_dependencyIdentity@gradients/discriminator/conv2d_2/Conv2D_grad/Conv2DBackpropInput>^gradients/discriminator/conv2d_2/Conv2D_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/discriminator/conv2d_2/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:?????????@
?
Ggradients/discriminator/conv2d_2/Conv2D_grad/tuple/control_dependency_1IdentityAgradients/discriminator/conv2d_2/Conv2D_grad/Conv2DBackpropFilter>^gradients/discriminator/conv2d_2/Conv2D_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients/discriminator/conv2d_2/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:@@
?
gradients/AddN_14AddNJgradients/discriminator_1/conv2d_2/BiasAdd_grad/tuple/control_dependency_1Hgradients/discriminator/conv2d_2/BiasAdd_grad/tuple/control_dependency_1*
T0*
N*N
_classD
B@loc:@gradients/discriminator_1/conv2d_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
?
5gradients/discriminator_1/conv2d_1/Relu_grad/ReluGradReluGradGgradients/discriminator_1/conv2d_2/Conv2D_grad/tuple/control_dependencydiscriminator_1/conv2d_1/Relu*
T0*/
_output_shapes
:?????????@
?
3gradients/discriminator/conv2d_1/Relu_grad/ReluGradReluGradEgradients/discriminator/conv2d_2/Conv2D_grad/tuple/control_dependencydiscriminator/conv2d_1/Relu*
T0*/
_output_shapes
:?????????@
?
gradients/AddN_15AddNIgradients/discriminator_1/conv2d_2/Conv2D_grad/tuple/control_dependency_1Ggradients/discriminator/conv2d_2/Conv2D_grad/tuple/control_dependency_1*
T0*
N*V
_classL
JHloc:@gradients/discriminator_1/conv2d_2/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:@@
?
;gradients/discriminator_1/conv2d_1/BiasAdd_grad/BiasAddGradBiasAddGrad5gradients/discriminator_1/conv2d_1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
?
@gradients/discriminator_1/conv2d_1/BiasAdd_grad/tuple/group_depsNoOp<^gradients/discriminator_1/conv2d_1/BiasAdd_grad/BiasAddGrad6^gradients/discriminator_1/conv2d_1/Relu_grad/ReluGrad
?
Hgradients/discriminator_1/conv2d_1/BiasAdd_grad/tuple/control_dependencyIdentity5gradients/discriminator_1/conv2d_1/Relu_grad/ReluGradA^gradients/discriminator_1/conv2d_1/BiasAdd_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/discriminator_1/conv2d_1/Relu_grad/ReluGrad*/
_output_shapes
:?????????@
?
Jgradients/discriminator_1/conv2d_1/BiasAdd_grad/tuple/control_dependency_1Identity;gradients/discriminator_1/conv2d_1/BiasAdd_grad/BiasAddGradA^gradients/discriminator_1/conv2d_1/BiasAdd_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients/discriminator_1/conv2d_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
?
9gradients/discriminator/conv2d_1/BiasAdd_grad/BiasAddGradBiasAddGrad3gradients/discriminator/conv2d_1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
?
>gradients/discriminator/conv2d_1/BiasAdd_grad/tuple/group_depsNoOp:^gradients/discriminator/conv2d_1/BiasAdd_grad/BiasAddGrad4^gradients/discriminator/conv2d_1/Relu_grad/ReluGrad
?
Fgradients/discriminator/conv2d_1/BiasAdd_grad/tuple/control_dependencyIdentity3gradients/discriminator/conv2d_1/Relu_grad/ReluGrad?^gradients/discriminator/conv2d_1/BiasAdd_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/discriminator/conv2d_1/Relu_grad/ReluGrad*/
_output_shapes
:?????????@
?
Hgradients/discriminator/conv2d_1/BiasAdd_grad/tuple/control_dependency_1Identity9gradients/discriminator/conv2d_1/BiasAdd_grad/BiasAddGrad?^gradients/discriminator/conv2d_1/BiasAdd_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/discriminator/conv2d_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
?
5gradients/discriminator_1/conv2d_1/Conv2D_grad/ShapeNShapeN"discriminator_1/dropout_1/Identity"discriminator/conv2d_1/kernel/read*
T0*
out_type0*
N* 
_output_shapes
::
?
Bgradients/discriminator_1/conv2d_1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput5gradients/discriminator_1/conv2d_1/Conv2D_grad/ShapeN"discriminator/conv2d_1/kernel/readHgradients/discriminator_1/conv2d_1/BiasAdd_grad/tuple/control_dependency*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(*/
_output_shapes
:?????????@
?
Cgradients/discriminator_1/conv2d_1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter"discriminator_1/dropout_1/Identity7gradients/discriminator_1/conv2d_1/Conv2D_grad/ShapeN:1Hgradients/discriminator_1/conv2d_1/BiasAdd_grad/tuple/control_dependency*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(*&
_output_shapes
:@@
?
?gradients/discriminator_1/conv2d_1/Conv2D_grad/tuple/group_depsNoOpD^gradients/discriminator_1/conv2d_1/Conv2D_grad/Conv2DBackpropFilterC^gradients/discriminator_1/conv2d_1/Conv2D_grad/Conv2DBackpropInput
?
Ggradients/discriminator_1/conv2d_1/Conv2D_grad/tuple/control_dependencyIdentityBgradients/discriminator_1/conv2d_1/Conv2D_grad/Conv2DBackpropInput@^gradients/discriminator_1/conv2d_1/Conv2D_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/discriminator_1/conv2d_1/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:?????????@
?
Igradients/discriminator_1/conv2d_1/Conv2D_grad/tuple/control_dependency_1IdentityCgradients/discriminator_1/conv2d_1/Conv2D_grad/Conv2DBackpropFilter@^gradients/discriminator_1/conv2d_1/Conv2D_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients/discriminator_1/conv2d_1/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:@@
?
3gradients/discriminator/conv2d_1/Conv2D_grad/ShapeNShapeN discriminator/dropout_1/Identity"discriminator/conv2d_1/kernel/read*
T0*
out_type0*
N* 
_output_shapes
::
?
@gradients/discriminator/conv2d_1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput3gradients/discriminator/conv2d_1/Conv2D_grad/ShapeN"discriminator/conv2d_1/kernel/readFgradients/discriminator/conv2d_1/BiasAdd_grad/tuple/control_dependency*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(*/
_output_shapes
:?????????@
?
Agradients/discriminator/conv2d_1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter discriminator/dropout_1/Identity5gradients/discriminator/conv2d_1/Conv2D_grad/ShapeN:1Fgradients/discriminator/conv2d_1/BiasAdd_grad/tuple/control_dependency*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(*&
_output_shapes
:@@
?
=gradients/discriminator/conv2d_1/Conv2D_grad/tuple/group_depsNoOpB^gradients/discriminator/conv2d_1/Conv2D_grad/Conv2DBackpropFilterA^gradients/discriminator/conv2d_1/Conv2D_grad/Conv2DBackpropInput
?
Egradients/discriminator/conv2d_1/Conv2D_grad/tuple/control_dependencyIdentity@gradients/discriminator/conv2d_1/Conv2D_grad/Conv2DBackpropInput>^gradients/discriminator/conv2d_1/Conv2D_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/discriminator/conv2d_1/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:?????????@
?
Ggradients/discriminator/conv2d_1/Conv2D_grad/tuple/control_dependency_1IdentityAgradients/discriminator/conv2d_1/Conv2D_grad/Conv2DBackpropFilter>^gradients/discriminator/conv2d_1/Conv2D_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients/discriminator/conv2d_1/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:@@
?
gradients/AddN_16AddNJgradients/discriminator_1/conv2d_1/BiasAdd_grad/tuple/control_dependency_1Hgradients/discriminator/conv2d_1/BiasAdd_grad/tuple/control_dependency_1*
T0*
N*N
_classD
B@loc:@gradients/discriminator_1/conv2d_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
?
gradients/AddN_17AddNIgradients/discriminator_1/conv2d_1/Conv2D_grad/tuple/control_dependency_1Ggradients/discriminator/conv2d_1/Conv2D_grad/tuple/control_dependency_1*
T0*
N*V
_classL
JHloc:@gradients/discriminator_1/conv2d_1/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:@@
?
3gradients/discriminator_1/conv2d/Relu_grad/ReluGradReluGradGgradients/discriminator_1/conv2d_1/Conv2D_grad/tuple/control_dependencydiscriminator_1/conv2d/Relu*
T0*/
_output_shapes
:?????????@
?
1gradients/discriminator/conv2d/Relu_grad/ReluGradReluGradEgradients/discriminator/conv2d_1/Conv2D_grad/tuple/control_dependencydiscriminator/conv2d/Relu*
T0*/
_output_shapes
:?????????@
?
9gradients/discriminator_1/conv2d/BiasAdd_grad/BiasAddGradBiasAddGrad3gradients/discriminator_1/conv2d/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
?
>gradients/discriminator_1/conv2d/BiasAdd_grad/tuple/group_depsNoOp:^gradients/discriminator_1/conv2d/BiasAdd_grad/BiasAddGrad4^gradients/discriminator_1/conv2d/Relu_grad/ReluGrad
?
Fgradients/discriminator_1/conv2d/BiasAdd_grad/tuple/control_dependencyIdentity3gradients/discriminator_1/conv2d/Relu_grad/ReluGrad?^gradients/discriminator_1/conv2d/BiasAdd_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/discriminator_1/conv2d/Relu_grad/ReluGrad*/
_output_shapes
:?????????@
?
Hgradients/discriminator_1/conv2d/BiasAdd_grad/tuple/control_dependency_1Identity9gradients/discriminator_1/conv2d/BiasAdd_grad/BiasAddGrad?^gradients/discriminator_1/conv2d/BiasAdd_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/discriminator_1/conv2d/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
?
7gradients/discriminator/conv2d/BiasAdd_grad/BiasAddGradBiasAddGrad1gradients/discriminator/conv2d/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
?
<gradients/discriminator/conv2d/BiasAdd_grad/tuple/group_depsNoOp8^gradients/discriminator/conv2d/BiasAdd_grad/BiasAddGrad2^gradients/discriminator/conv2d/Relu_grad/ReluGrad
?
Dgradients/discriminator/conv2d/BiasAdd_grad/tuple/control_dependencyIdentity1gradients/discriminator/conv2d/Relu_grad/ReluGrad=^gradients/discriminator/conv2d/BiasAdd_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/discriminator/conv2d/Relu_grad/ReluGrad*/
_output_shapes
:?????????@
?
Fgradients/discriminator/conv2d/BiasAdd_grad/tuple/control_dependency_1Identity7gradients/discriminator/conv2d/BiasAdd_grad/BiasAddGrad=^gradients/discriminator/conv2d/BiasAdd_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/discriminator/conv2d/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
?
3gradients/discriminator_1/conv2d/Conv2D_grad/ShapeNShapeN discriminator_1/dropout/Identity discriminator/conv2d/kernel/read*
T0*
out_type0*
N* 
_output_shapes
::
?
@gradients/discriminator_1/conv2d/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput3gradients/discriminator_1/conv2d/Conv2D_grad/ShapeN discriminator/conv2d/kernel/readFgradients/discriminator_1/conv2d/BiasAdd_grad/tuple/control_dependency*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(*/
_output_shapes
:?????????  
?
Agradients/discriminator_1/conv2d/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter discriminator_1/dropout/Identity5gradients/discriminator_1/conv2d/Conv2D_grad/ShapeN:1Fgradients/discriminator_1/conv2d/BiasAdd_grad/tuple/control_dependency*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(*&
_output_shapes
:@
?
=gradients/discriminator_1/conv2d/Conv2D_grad/tuple/group_depsNoOpB^gradients/discriminator_1/conv2d/Conv2D_grad/Conv2DBackpropFilterA^gradients/discriminator_1/conv2d/Conv2D_grad/Conv2DBackpropInput
?
Egradients/discriminator_1/conv2d/Conv2D_grad/tuple/control_dependencyIdentity@gradients/discriminator_1/conv2d/Conv2D_grad/Conv2DBackpropInput>^gradients/discriminator_1/conv2d/Conv2D_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/discriminator_1/conv2d/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:?????????  
?
Ggradients/discriminator_1/conv2d/Conv2D_grad/tuple/control_dependency_1IdentityAgradients/discriminator_1/conv2d/Conv2D_grad/Conv2DBackpropFilter>^gradients/discriminator_1/conv2d/Conv2D_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients/discriminator_1/conv2d/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:@
?
1gradients/discriminator/conv2d/Conv2D_grad/ShapeNShapeNdiscriminator/dropout/Identity discriminator/conv2d/kernel/read*
T0*
out_type0*
N* 
_output_shapes
::
?
>gradients/discriminator/conv2d/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput1gradients/discriminator/conv2d/Conv2D_grad/ShapeN discriminator/conv2d/kernel/readDgradients/discriminator/conv2d/BiasAdd_grad/tuple/control_dependency*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(*/
_output_shapes
:?????????  
?
?gradients/discriminator/conv2d/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterdiscriminator/dropout/Identity3gradients/discriminator/conv2d/Conv2D_grad/ShapeN:1Dgradients/discriminator/conv2d/BiasAdd_grad/tuple/control_dependency*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(*&
_output_shapes
:@
?
;gradients/discriminator/conv2d/Conv2D_grad/tuple/group_depsNoOp@^gradients/discriminator/conv2d/Conv2D_grad/Conv2DBackpropFilter?^gradients/discriminator/conv2d/Conv2D_grad/Conv2DBackpropInput
?
Cgradients/discriminator/conv2d/Conv2D_grad/tuple/control_dependencyIdentity>gradients/discriminator/conv2d/Conv2D_grad/Conv2DBackpropInput<^gradients/discriminator/conv2d/Conv2D_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/discriminator/conv2d/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:?????????  
?
Egradients/discriminator/conv2d/Conv2D_grad/tuple/control_dependency_1Identity?gradients/discriminator/conv2d/Conv2D_grad/Conv2DBackpropFilter<^gradients/discriminator/conv2d/Conv2D_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients/discriminator/conv2d/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:@
?
gradients/AddN_18AddNHgradients/discriminator_1/conv2d/BiasAdd_grad/tuple/control_dependency_1Fgradients/discriminator/conv2d/BiasAdd_grad/tuple/control_dependency_1*
T0*
N*L
_classB
@>loc:@gradients/discriminator_1/conv2d/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
?
gradients/AddN_19AddNGgradients/discriminator_1/conv2d/Conv2D_grad/tuple/control_dependency_1Egradients/discriminator/conv2d/Conv2D_grad/tuple/control_dependency_1*
T0*
N*T
_classJ
HFloc:@gradients/discriminator_1/conv2d/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:@
?
beta1_power/initial_valueConst*
valueB
 *   ?*
dtype0*,
_class"
 loc:@discriminator/conv2d/bias*
_output_shapes
: 
?
beta1_power
VariableV2*
dtype0*
shared_name *
shape: *
	container *,
_class"
 loc:@discriminator/conv2d/bias*
_output_shapes
: 
?
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
T0*
use_locking(*
validate_shape(*,
_class"
 loc:@discriminator/conv2d/bias*
_output_shapes
: 
x
beta1_power/readIdentitybeta1_power*
T0*,
_class"
 loc:@discriminator/conv2d/bias*
_output_shapes
: 
?
beta2_power/initial_valueConst*
valueB
 *w??*
dtype0*,
_class"
 loc:@discriminator/conv2d/bias*
_output_shapes
: 
?
beta2_power
VariableV2*
dtype0*
shared_name *
shape: *
	container *,
_class"
 loc:@discriminator/conv2d/bias*
_output_shapes
: 
?
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
T0*
use_locking(*
validate_shape(*,
_class"
 loc:@discriminator/conv2d/bias*
_output_shapes
: 
x
beta2_power/readIdentitybeta2_power*
T0*,
_class"
 loc:@discriminator/conv2d/bias*
_output_shapes
: 
?
2discriminator/conv2d/kernel/Adam/Initializer/zerosConst*%
valueB@*    *
dtype0*.
_class$
" loc:@discriminator/conv2d/kernel*&
_output_shapes
:@
?
 discriminator/conv2d/kernel/Adam
VariableV2*
dtype0*
shared_name *
shape:@*
	container *.
_class$
" loc:@discriminator/conv2d/kernel*&
_output_shapes
:@
?
'discriminator/conv2d/kernel/Adam/AssignAssign discriminator/conv2d/kernel/Adam2discriminator/conv2d/kernel/Adam/Initializer/zeros*
T0*
use_locking(*
validate_shape(*.
_class$
" loc:@discriminator/conv2d/kernel*&
_output_shapes
:@
?
%discriminator/conv2d/kernel/Adam/readIdentity discriminator/conv2d/kernel/Adam*
T0*.
_class$
" loc:@discriminator/conv2d/kernel*&
_output_shapes
:@
?
4discriminator/conv2d/kernel/Adam_1/Initializer/zerosConst*%
valueB@*    *
dtype0*.
_class$
" loc:@discriminator/conv2d/kernel*&
_output_shapes
:@
?
"discriminator/conv2d/kernel/Adam_1
VariableV2*
dtype0*
shared_name *
shape:@*
	container *.
_class$
" loc:@discriminator/conv2d/kernel*&
_output_shapes
:@
?
)discriminator/conv2d/kernel/Adam_1/AssignAssign"discriminator/conv2d/kernel/Adam_14discriminator/conv2d/kernel/Adam_1/Initializer/zeros*
T0*
use_locking(*
validate_shape(*.
_class$
" loc:@discriminator/conv2d/kernel*&
_output_shapes
:@
?
'discriminator/conv2d/kernel/Adam_1/readIdentity"discriminator/conv2d/kernel/Adam_1*
T0*.
_class$
" loc:@discriminator/conv2d/kernel*&
_output_shapes
:@
?
0discriminator/conv2d/bias/Adam/Initializer/zerosConst*
valueB@*    *
dtype0*,
_class"
 loc:@discriminator/conv2d/bias*
_output_shapes
:@
?
discriminator/conv2d/bias/Adam
VariableV2*
dtype0*
shared_name *
shape:@*
	container *,
_class"
 loc:@discriminator/conv2d/bias*
_output_shapes
:@
?
%discriminator/conv2d/bias/Adam/AssignAssigndiscriminator/conv2d/bias/Adam0discriminator/conv2d/bias/Adam/Initializer/zeros*
T0*
use_locking(*
validate_shape(*,
_class"
 loc:@discriminator/conv2d/bias*
_output_shapes
:@
?
#discriminator/conv2d/bias/Adam/readIdentitydiscriminator/conv2d/bias/Adam*
T0*,
_class"
 loc:@discriminator/conv2d/bias*
_output_shapes
:@
?
2discriminator/conv2d/bias/Adam_1/Initializer/zerosConst*
valueB@*    *
dtype0*,
_class"
 loc:@discriminator/conv2d/bias*
_output_shapes
:@
?
 discriminator/conv2d/bias/Adam_1
VariableV2*
dtype0*
shared_name *
shape:@*
	container *,
_class"
 loc:@discriminator/conv2d/bias*
_output_shapes
:@
?
'discriminator/conv2d/bias/Adam_1/AssignAssign discriminator/conv2d/bias/Adam_12discriminator/conv2d/bias/Adam_1/Initializer/zeros*
T0*
use_locking(*
validate_shape(*,
_class"
 loc:@discriminator/conv2d/bias*
_output_shapes
:@
?
%discriminator/conv2d/bias/Adam_1/readIdentity discriminator/conv2d/bias/Adam_1*
T0*,
_class"
 loc:@discriminator/conv2d/bias*
_output_shapes
:@
?
Ddiscriminator/conv2d_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"      @   @   *
dtype0*0
_class&
$"loc:@discriminator/conv2d_1/kernel*
_output_shapes
:
?
:discriminator/conv2d_1/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*0
_class&
$"loc:@discriminator/conv2d_1/kernel*
_output_shapes
: 
?
4discriminator/conv2d_1/kernel/Adam/Initializer/zerosFillDdiscriminator/conv2d_1/kernel/Adam/Initializer/zeros/shape_as_tensor:discriminator/conv2d_1/kernel/Adam/Initializer/zeros/Const*
T0*

index_type0*0
_class&
$"loc:@discriminator/conv2d_1/kernel*&
_output_shapes
:@@
?
"discriminator/conv2d_1/kernel/Adam
VariableV2*
dtype0*
shared_name *
shape:@@*
	container *0
_class&
$"loc:@discriminator/conv2d_1/kernel*&
_output_shapes
:@@
?
)discriminator/conv2d_1/kernel/Adam/AssignAssign"discriminator/conv2d_1/kernel/Adam4discriminator/conv2d_1/kernel/Adam/Initializer/zeros*
T0*
use_locking(*
validate_shape(*0
_class&
$"loc:@discriminator/conv2d_1/kernel*&
_output_shapes
:@@
?
'discriminator/conv2d_1/kernel/Adam/readIdentity"discriminator/conv2d_1/kernel/Adam*
T0*0
_class&
$"loc:@discriminator/conv2d_1/kernel*&
_output_shapes
:@@
?
Fdiscriminator/conv2d_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"      @   @   *
dtype0*0
_class&
$"loc:@discriminator/conv2d_1/kernel*
_output_shapes
:
?
<discriminator/conv2d_1/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*0
_class&
$"loc:@discriminator/conv2d_1/kernel*
_output_shapes
: 
?
6discriminator/conv2d_1/kernel/Adam_1/Initializer/zerosFillFdiscriminator/conv2d_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor<discriminator/conv2d_1/kernel/Adam_1/Initializer/zeros/Const*
T0*

index_type0*0
_class&
$"loc:@discriminator/conv2d_1/kernel*&
_output_shapes
:@@
?
$discriminator/conv2d_1/kernel/Adam_1
VariableV2*
dtype0*
shared_name *
shape:@@*
	container *0
_class&
$"loc:@discriminator/conv2d_1/kernel*&
_output_shapes
:@@
?
+discriminator/conv2d_1/kernel/Adam_1/AssignAssign$discriminator/conv2d_1/kernel/Adam_16discriminator/conv2d_1/kernel/Adam_1/Initializer/zeros*
T0*
use_locking(*
validate_shape(*0
_class&
$"loc:@discriminator/conv2d_1/kernel*&
_output_shapes
:@@
?
)discriminator/conv2d_1/kernel/Adam_1/readIdentity$discriminator/conv2d_1/kernel/Adam_1*
T0*0
_class&
$"loc:@discriminator/conv2d_1/kernel*&
_output_shapes
:@@
?
2discriminator/conv2d_1/bias/Adam/Initializer/zerosConst*
valueB@*    *
dtype0*.
_class$
" loc:@discriminator/conv2d_1/bias*
_output_shapes
:@
?
 discriminator/conv2d_1/bias/Adam
VariableV2*
dtype0*
shared_name *
shape:@*
	container *.
_class$
" loc:@discriminator/conv2d_1/bias*
_output_shapes
:@
?
'discriminator/conv2d_1/bias/Adam/AssignAssign discriminator/conv2d_1/bias/Adam2discriminator/conv2d_1/bias/Adam/Initializer/zeros*
T0*
use_locking(*
validate_shape(*.
_class$
" loc:@discriminator/conv2d_1/bias*
_output_shapes
:@
?
%discriminator/conv2d_1/bias/Adam/readIdentity discriminator/conv2d_1/bias/Adam*
T0*.
_class$
" loc:@discriminator/conv2d_1/bias*
_output_shapes
:@
?
4discriminator/conv2d_1/bias/Adam_1/Initializer/zerosConst*
valueB@*    *
dtype0*.
_class$
" loc:@discriminator/conv2d_1/bias*
_output_shapes
:@
?
"discriminator/conv2d_1/bias/Adam_1
VariableV2*
dtype0*
shared_name *
shape:@*
	container *.
_class$
" loc:@discriminator/conv2d_1/bias*
_output_shapes
:@
?
)discriminator/conv2d_1/bias/Adam_1/AssignAssign"discriminator/conv2d_1/bias/Adam_14discriminator/conv2d_1/bias/Adam_1/Initializer/zeros*
T0*
use_locking(*
validate_shape(*.
_class$
" loc:@discriminator/conv2d_1/bias*
_output_shapes
:@
?
'discriminator/conv2d_1/bias/Adam_1/readIdentity"discriminator/conv2d_1/bias/Adam_1*
T0*.
_class$
" loc:@discriminator/conv2d_1/bias*
_output_shapes
:@
?
Ddiscriminator/conv2d_2/kernel/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"      @   @   *
dtype0*0
_class&
$"loc:@discriminator/conv2d_2/kernel*
_output_shapes
:
?
:discriminator/conv2d_2/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*0
_class&
$"loc:@discriminator/conv2d_2/kernel*
_output_shapes
: 
?
4discriminator/conv2d_2/kernel/Adam/Initializer/zerosFillDdiscriminator/conv2d_2/kernel/Adam/Initializer/zeros/shape_as_tensor:discriminator/conv2d_2/kernel/Adam/Initializer/zeros/Const*
T0*

index_type0*0
_class&
$"loc:@discriminator/conv2d_2/kernel*&
_output_shapes
:@@
?
"discriminator/conv2d_2/kernel/Adam
VariableV2*
dtype0*
shared_name *
shape:@@*
	container *0
_class&
$"loc:@discriminator/conv2d_2/kernel*&
_output_shapes
:@@
?
)discriminator/conv2d_2/kernel/Adam/AssignAssign"discriminator/conv2d_2/kernel/Adam4discriminator/conv2d_2/kernel/Adam/Initializer/zeros*
T0*
use_locking(*
validate_shape(*0
_class&
$"loc:@discriminator/conv2d_2/kernel*&
_output_shapes
:@@
?
'discriminator/conv2d_2/kernel/Adam/readIdentity"discriminator/conv2d_2/kernel/Adam*
T0*0
_class&
$"loc:@discriminator/conv2d_2/kernel*&
_output_shapes
:@@
?
Fdiscriminator/conv2d_2/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"      @   @   *
dtype0*0
_class&
$"loc:@discriminator/conv2d_2/kernel*
_output_shapes
:
?
<discriminator/conv2d_2/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*0
_class&
$"loc:@discriminator/conv2d_2/kernel*
_output_shapes
: 
?
6discriminator/conv2d_2/kernel/Adam_1/Initializer/zerosFillFdiscriminator/conv2d_2/kernel/Adam_1/Initializer/zeros/shape_as_tensor<discriminator/conv2d_2/kernel/Adam_1/Initializer/zeros/Const*
T0*

index_type0*0
_class&
$"loc:@discriminator/conv2d_2/kernel*&
_output_shapes
:@@
?
$discriminator/conv2d_2/kernel/Adam_1
VariableV2*
dtype0*
shared_name *
shape:@@*
	container *0
_class&
$"loc:@discriminator/conv2d_2/kernel*&
_output_shapes
:@@
?
+discriminator/conv2d_2/kernel/Adam_1/AssignAssign$discriminator/conv2d_2/kernel/Adam_16discriminator/conv2d_2/kernel/Adam_1/Initializer/zeros*
T0*
use_locking(*
validate_shape(*0
_class&
$"loc:@discriminator/conv2d_2/kernel*&
_output_shapes
:@@
?
)discriminator/conv2d_2/kernel/Adam_1/readIdentity$discriminator/conv2d_2/kernel/Adam_1*
T0*0
_class&
$"loc:@discriminator/conv2d_2/kernel*&
_output_shapes
:@@
?
2discriminator/conv2d_2/bias/Adam/Initializer/zerosConst*
valueB@*    *
dtype0*.
_class$
" loc:@discriminator/conv2d_2/bias*
_output_shapes
:@
?
 discriminator/conv2d_2/bias/Adam
VariableV2*
dtype0*
shared_name *
shape:@*
	container *.
_class$
" loc:@discriminator/conv2d_2/bias*
_output_shapes
:@
?
'discriminator/conv2d_2/bias/Adam/AssignAssign discriminator/conv2d_2/bias/Adam2discriminator/conv2d_2/bias/Adam/Initializer/zeros*
T0*
use_locking(*
validate_shape(*.
_class$
" loc:@discriminator/conv2d_2/bias*
_output_shapes
:@
?
%discriminator/conv2d_2/bias/Adam/readIdentity discriminator/conv2d_2/bias/Adam*
T0*.
_class$
" loc:@discriminator/conv2d_2/bias*
_output_shapes
:@
?
4discriminator/conv2d_2/bias/Adam_1/Initializer/zerosConst*
valueB@*    *
dtype0*.
_class$
" loc:@discriminator/conv2d_2/bias*
_output_shapes
:@
?
"discriminator/conv2d_2/bias/Adam_1
VariableV2*
dtype0*
shared_name *
shape:@*
	container *.
_class$
" loc:@discriminator/conv2d_2/bias*
_output_shapes
:@
?
)discriminator/conv2d_2/bias/Adam_1/AssignAssign"discriminator/conv2d_2/bias/Adam_14discriminator/conv2d_2/bias/Adam_1/Initializer/zeros*
T0*
use_locking(*
validate_shape(*.
_class$
" loc:@discriminator/conv2d_2/bias*
_output_shapes
:@
?
'discriminator/conv2d_2/bias/Adam_1/readIdentity"discriminator/conv2d_2/bias/Adam_1*
T0*.
_class$
" loc:@discriminator/conv2d_2/bias*
_output_shapes
:@
?
Ddiscriminator/conv2d_3/kernel/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"      @   ?   *
dtype0*0
_class&
$"loc:@discriminator/conv2d_3/kernel*
_output_shapes
:
?
:discriminator/conv2d_3/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*0
_class&
$"loc:@discriminator/conv2d_3/kernel*
_output_shapes
: 
?
4discriminator/conv2d_3/kernel/Adam/Initializer/zerosFillDdiscriminator/conv2d_3/kernel/Adam/Initializer/zeros/shape_as_tensor:discriminator/conv2d_3/kernel/Adam/Initializer/zeros/Const*
T0*

index_type0*0
_class&
$"loc:@discriminator/conv2d_3/kernel*'
_output_shapes
:@?
?
"discriminator/conv2d_3/kernel/Adam
VariableV2*
dtype0*
shared_name *
shape:@?*
	container *0
_class&
$"loc:@discriminator/conv2d_3/kernel*'
_output_shapes
:@?
?
)discriminator/conv2d_3/kernel/Adam/AssignAssign"discriminator/conv2d_3/kernel/Adam4discriminator/conv2d_3/kernel/Adam/Initializer/zeros*
T0*
use_locking(*
validate_shape(*0
_class&
$"loc:@discriminator/conv2d_3/kernel*'
_output_shapes
:@?
?
'discriminator/conv2d_3/kernel/Adam/readIdentity"discriminator/conv2d_3/kernel/Adam*
T0*0
_class&
$"loc:@discriminator/conv2d_3/kernel*'
_output_shapes
:@?
?
Fdiscriminator/conv2d_3/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"      @   ?   *
dtype0*0
_class&
$"loc:@discriminator/conv2d_3/kernel*
_output_shapes
:
?
<discriminator/conv2d_3/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*0
_class&
$"loc:@discriminator/conv2d_3/kernel*
_output_shapes
: 
?
6discriminator/conv2d_3/kernel/Adam_1/Initializer/zerosFillFdiscriminator/conv2d_3/kernel/Adam_1/Initializer/zeros/shape_as_tensor<discriminator/conv2d_3/kernel/Adam_1/Initializer/zeros/Const*
T0*

index_type0*0
_class&
$"loc:@discriminator/conv2d_3/kernel*'
_output_shapes
:@?
?
$discriminator/conv2d_3/kernel/Adam_1
VariableV2*
dtype0*
shared_name *
shape:@?*
	container *0
_class&
$"loc:@discriminator/conv2d_3/kernel*'
_output_shapes
:@?
?
+discriminator/conv2d_3/kernel/Adam_1/AssignAssign$discriminator/conv2d_3/kernel/Adam_16discriminator/conv2d_3/kernel/Adam_1/Initializer/zeros*
T0*
use_locking(*
validate_shape(*0
_class&
$"loc:@discriminator/conv2d_3/kernel*'
_output_shapes
:@?
?
)discriminator/conv2d_3/kernel/Adam_1/readIdentity$discriminator/conv2d_3/kernel/Adam_1*
T0*0
_class&
$"loc:@discriminator/conv2d_3/kernel*'
_output_shapes
:@?
?
2discriminator/conv2d_3/bias/Adam/Initializer/zerosConst*
valueB?*    *
dtype0*.
_class$
" loc:@discriminator/conv2d_3/bias*
_output_shapes	
:?
?
 discriminator/conv2d_3/bias/Adam
VariableV2*
dtype0*
shared_name *
shape:?*
	container *.
_class$
" loc:@discriminator/conv2d_3/bias*
_output_shapes	
:?
?
'discriminator/conv2d_3/bias/Adam/AssignAssign discriminator/conv2d_3/bias/Adam2discriminator/conv2d_3/bias/Adam/Initializer/zeros*
T0*
use_locking(*
validate_shape(*.
_class$
" loc:@discriminator/conv2d_3/bias*
_output_shapes	
:?
?
%discriminator/conv2d_3/bias/Adam/readIdentity discriminator/conv2d_3/bias/Adam*
T0*.
_class$
" loc:@discriminator/conv2d_3/bias*
_output_shapes	
:?
?
4discriminator/conv2d_3/bias/Adam_1/Initializer/zerosConst*
valueB?*    *
dtype0*.
_class$
" loc:@discriminator/conv2d_3/bias*
_output_shapes	
:?
?
"discriminator/conv2d_3/bias/Adam_1
VariableV2*
dtype0*
shared_name *
shape:?*
	container *.
_class$
" loc:@discriminator/conv2d_3/bias*
_output_shapes	
:?
?
)discriminator/conv2d_3/bias/Adam_1/AssignAssign"discriminator/conv2d_3/bias/Adam_14discriminator/conv2d_3/bias/Adam_1/Initializer/zeros*
T0*
use_locking(*
validate_shape(*.
_class$
" loc:@discriminator/conv2d_3/bias*
_output_shapes	
:?
?
'discriminator/conv2d_3/bias/Adam_1/readIdentity"discriminator/conv2d_3/bias/Adam_1*
T0*.
_class$
" loc:@discriminator/conv2d_3/bias*
_output_shapes	
:?
?
Ddiscriminator/conv2d_4/kernel/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"      ?   ?   *
dtype0*0
_class&
$"loc:@discriminator/conv2d_4/kernel*
_output_shapes
:
?
:discriminator/conv2d_4/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*0
_class&
$"loc:@discriminator/conv2d_4/kernel*
_output_shapes
: 
?
4discriminator/conv2d_4/kernel/Adam/Initializer/zerosFillDdiscriminator/conv2d_4/kernel/Adam/Initializer/zeros/shape_as_tensor:discriminator/conv2d_4/kernel/Adam/Initializer/zeros/Const*
T0*

index_type0*0
_class&
$"loc:@discriminator/conv2d_4/kernel*(
_output_shapes
:??
?
"discriminator/conv2d_4/kernel/Adam
VariableV2*
dtype0*
shared_name *
shape:??*
	container *0
_class&
$"loc:@discriminator/conv2d_4/kernel*(
_output_shapes
:??
?
)discriminator/conv2d_4/kernel/Adam/AssignAssign"discriminator/conv2d_4/kernel/Adam4discriminator/conv2d_4/kernel/Adam/Initializer/zeros*
T0*
use_locking(*
validate_shape(*0
_class&
$"loc:@discriminator/conv2d_4/kernel*(
_output_shapes
:??
?
'discriminator/conv2d_4/kernel/Adam/readIdentity"discriminator/conv2d_4/kernel/Adam*
T0*0
_class&
$"loc:@discriminator/conv2d_4/kernel*(
_output_shapes
:??
?
Fdiscriminator/conv2d_4/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"      ?   ?   *
dtype0*0
_class&
$"loc:@discriminator/conv2d_4/kernel*
_output_shapes
:
?
<discriminator/conv2d_4/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*0
_class&
$"loc:@discriminator/conv2d_4/kernel*
_output_shapes
: 
?
6discriminator/conv2d_4/kernel/Adam_1/Initializer/zerosFillFdiscriminator/conv2d_4/kernel/Adam_1/Initializer/zeros/shape_as_tensor<discriminator/conv2d_4/kernel/Adam_1/Initializer/zeros/Const*
T0*

index_type0*0
_class&
$"loc:@discriminator/conv2d_4/kernel*(
_output_shapes
:??
?
$discriminator/conv2d_4/kernel/Adam_1
VariableV2*
dtype0*
shared_name *
shape:??*
	container *0
_class&
$"loc:@discriminator/conv2d_4/kernel*(
_output_shapes
:??
?
+discriminator/conv2d_4/kernel/Adam_1/AssignAssign$discriminator/conv2d_4/kernel/Adam_16discriminator/conv2d_4/kernel/Adam_1/Initializer/zeros*
T0*
use_locking(*
validate_shape(*0
_class&
$"loc:@discriminator/conv2d_4/kernel*(
_output_shapes
:??
?
)discriminator/conv2d_4/kernel/Adam_1/readIdentity$discriminator/conv2d_4/kernel/Adam_1*
T0*0
_class&
$"loc:@discriminator/conv2d_4/kernel*(
_output_shapes
:??
?
2discriminator/conv2d_4/bias/Adam/Initializer/zerosConst*
valueB?*    *
dtype0*.
_class$
" loc:@discriminator/conv2d_4/bias*
_output_shapes	
:?
?
 discriminator/conv2d_4/bias/Adam
VariableV2*
dtype0*
shared_name *
shape:?*
	container *.
_class$
" loc:@discriminator/conv2d_4/bias*
_output_shapes	
:?
?
'discriminator/conv2d_4/bias/Adam/AssignAssign discriminator/conv2d_4/bias/Adam2discriminator/conv2d_4/bias/Adam/Initializer/zeros*
T0*
use_locking(*
validate_shape(*.
_class$
" loc:@discriminator/conv2d_4/bias*
_output_shapes	
:?
?
%discriminator/conv2d_4/bias/Adam/readIdentity discriminator/conv2d_4/bias/Adam*
T0*.
_class$
" loc:@discriminator/conv2d_4/bias*
_output_shapes	
:?
?
4discriminator/conv2d_4/bias/Adam_1/Initializer/zerosConst*
valueB?*    *
dtype0*.
_class$
" loc:@discriminator/conv2d_4/bias*
_output_shapes	
:?
?
"discriminator/conv2d_4/bias/Adam_1
VariableV2*
dtype0*
shared_name *
shape:?*
	container *.
_class$
" loc:@discriminator/conv2d_4/bias*
_output_shapes	
:?
?
)discriminator/conv2d_4/bias/Adam_1/AssignAssign"discriminator/conv2d_4/bias/Adam_14discriminator/conv2d_4/bias/Adam_1/Initializer/zeros*
T0*
use_locking(*
validate_shape(*.
_class$
" loc:@discriminator/conv2d_4/bias*
_output_shapes	
:?
?
'discriminator/conv2d_4/bias/Adam_1/readIdentity"discriminator/conv2d_4/bias/Adam_1*
T0*.
_class$
" loc:@discriminator/conv2d_4/bias*
_output_shapes	
:?
?
Ddiscriminator/conv2d_5/kernel/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"      ?   ?   *
dtype0*0
_class&
$"loc:@discriminator/conv2d_5/kernel*
_output_shapes
:
?
:discriminator/conv2d_5/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*0
_class&
$"loc:@discriminator/conv2d_5/kernel*
_output_shapes
: 
?
4discriminator/conv2d_5/kernel/Adam/Initializer/zerosFillDdiscriminator/conv2d_5/kernel/Adam/Initializer/zeros/shape_as_tensor:discriminator/conv2d_5/kernel/Adam/Initializer/zeros/Const*
T0*

index_type0*0
_class&
$"loc:@discriminator/conv2d_5/kernel*(
_output_shapes
:??
?
"discriminator/conv2d_5/kernel/Adam
VariableV2*
dtype0*
shared_name *
shape:??*
	container *0
_class&
$"loc:@discriminator/conv2d_5/kernel*(
_output_shapes
:??
?
)discriminator/conv2d_5/kernel/Adam/AssignAssign"discriminator/conv2d_5/kernel/Adam4discriminator/conv2d_5/kernel/Adam/Initializer/zeros*
T0*
use_locking(*
validate_shape(*0
_class&
$"loc:@discriminator/conv2d_5/kernel*(
_output_shapes
:??
?
'discriminator/conv2d_5/kernel/Adam/readIdentity"discriminator/conv2d_5/kernel/Adam*
T0*0
_class&
$"loc:@discriminator/conv2d_5/kernel*(
_output_shapes
:??
?
Fdiscriminator/conv2d_5/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"      ?   ?   *
dtype0*0
_class&
$"loc:@discriminator/conv2d_5/kernel*
_output_shapes
:
?
<discriminator/conv2d_5/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*0
_class&
$"loc:@discriminator/conv2d_5/kernel*
_output_shapes
: 
?
6discriminator/conv2d_5/kernel/Adam_1/Initializer/zerosFillFdiscriminator/conv2d_5/kernel/Adam_1/Initializer/zeros/shape_as_tensor<discriminator/conv2d_5/kernel/Adam_1/Initializer/zeros/Const*
T0*

index_type0*0
_class&
$"loc:@discriminator/conv2d_5/kernel*(
_output_shapes
:??
?
$discriminator/conv2d_5/kernel/Adam_1
VariableV2*
dtype0*
shared_name *
shape:??*
	container *0
_class&
$"loc:@discriminator/conv2d_5/kernel*(
_output_shapes
:??
?
+discriminator/conv2d_5/kernel/Adam_1/AssignAssign$discriminator/conv2d_5/kernel/Adam_16discriminator/conv2d_5/kernel/Adam_1/Initializer/zeros*
T0*
use_locking(*
validate_shape(*0
_class&
$"loc:@discriminator/conv2d_5/kernel*(
_output_shapes
:??
?
)discriminator/conv2d_5/kernel/Adam_1/readIdentity$discriminator/conv2d_5/kernel/Adam_1*
T0*0
_class&
$"loc:@discriminator/conv2d_5/kernel*(
_output_shapes
:??
?
2discriminator/conv2d_5/bias/Adam/Initializer/zerosConst*
valueB?*    *
dtype0*.
_class$
" loc:@discriminator/conv2d_5/bias*
_output_shapes	
:?
?
 discriminator/conv2d_5/bias/Adam
VariableV2*
dtype0*
shared_name *
shape:?*
	container *.
_class$
" loc:@discriminator/conv2d_5/bias*
_output_shapes	
:?
?
'discriminator/conv2d_5/bias/Adam/AssignAssign discriminator/conv2d_5/bias/Adam2discriminator/conv2d_5/bias/Adam/Initializer/zeros*
T0*
use_locking(*
validate_shape(*.
_class$
" loc:@discriminator/conv2d_5/bias*
_output_shapes	
:?
?
%discriminator/conv2d_5/bias/Adam/readIdentity discriminator/conv2d_5/bias/Adam*
T0*.
_class$
" loc:@discriminator/conv2d_5/bias*
_output_shapes	
:?
?
4discriminator/conv2d_5/bias/Adam_1/Initializer/zerosConst*
valueB?*    *
dtype0*.
_class$
" loc:@discriminator/conv2d_5/bias*
_output_shapes	
:?
?
"discriminator/conv2d_5/bias/Adam_1
VariableV2*
dtype0*
shared_name *
shape:?*
	container *.
_class$
" loc:@discriminator/conv2d_5/bias*
_output_shapes	
:?
?
)discriminator/conv2d_5/bias/Adam_1/AssignAssign"discriminator/conv2d_5/bias/Adam_14discriminator/conv2d_5/bias/Adam_1/Initializer/zeros*
T0*
use_locking(*
validate_shape(*.
_class$
" loc:@discriminator/conv2d_5/bias*
_output_shapes	
:?
?
'discriminator/conv2d_5/bias/Adam_1/readIdentity"discriminator/conv2d_5/bias/Adam_1*
T0*.
_class$
" loc:@discriminator/conv2d_5/bias*
_output_shapes	
:?
?
Adiscriminator/dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"?   ?   *
dtype0*-
_class#
!loc:@discriminator/dense/kernel*
_output_shapes
:
?
7discriminator/dense/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*-
_class#
!loc:@discriminator/dense/kernel*
_output_shapes
: 
?
1discriminator/dense/kernel/Adam/Initializer/zerosFillAdiscriminator/dense/kernel/Adam/Initializer/zeros/shape_as_tensor7discriminator/dense/kernel/Adam/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@discriminator/dense/kernel*
_output_shapes
:	??
?
discriminator/dense/kernel/Adam
VariableV2*
dtype0*
shared_name *
shape:	??*
	container *-
_class#
!loc:@discriminator/dense/kernel*
_output_shapes
:	??
?
&discriminator/dense/kernel/Adam/AssignAssigndiscriminator/dense/kernel/Adam1discriminator/dense/kernel/Adam/Initializer/zeros*
T0*
use_locking(*
validate_shape(*-
_class#
!loc:@discriminator/dense/kernel*
_output_shapes
:	??
?
$discriminator/dense/kernel/Adam/readIdentitydiscriminator/dense/kernel/Adam*
T0*-
_class#
!loc:@discriminator/dense/kernel*
_output_shapes
:	??
?
Cdiscriminator/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"?   ?   *
dtype0*-
_class#
!loc:@discriminator/dense/kernel*
_output_shapes
:
?
9discriminator/dense/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*-
_class#
!loc:@discriminator/dense/kernel*
_output_shapes
: 
?
3discriminator/dense/kernel/Adam_1/Initializer/zerosFillCdiscriminator/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor9discriminator/dense/kernel/Adam_1/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@discriminator/dense/kernel*
_output_shapes
:	??
?
!discriminator/dense/kernel/Adam_1
VariableV2*
dtype0*
shared_name *
shape:	??*
	container *-
_class#
!loc:@discriminator/dense/kernel*
_output_shapes
:	??
?
(discriminator/dense/kernel/Adam_1/AssignAssign!discriminator/dense/kernel/Adam_13discriminator/dense/kernel/Adam_1/Initializer/zeros*
T0*
use_locking(*
validate_shape(*-
_class#
!loc:@discriminator/dense/kernel*
_output_shapes
:	??
?
&discriminator/dense/kernel/Adam_1/readIdentity!discriminator/dense/kernel/Adam_1*
T0*-
_class#
!loc:@discriminator/dense/kernel*
_output_shapes
:	??
?
/discriminator/dense/bias/Adam/Initializer/zerosConst*
valueB?*    *
dtype0*+
_class!
loc:@discriminator/dense/bias*
_output_shapes
:?
?
discriminator/dense/bias/Adam
VariableV2*
dtype0*
shared_name *
shape:?*
	container *+
_class!
loc:@discriminator/dense/bias*
_output_shapes
:?
?
$discriminator/dense/bias/Adam/AssignAssigndiscriminator/dense/bias/Adam/discriminator/dense/bias/Adam/Initializer/zeros*
T0*
use_locking(*
validate_shape(*+
_class!
loc:@discriminator/dense/bias*
_output_shapes
:?
?
"discriminator/dense/bias/Adam/readIdentitydiscriminator/dense/bias/Adam*
T0*+
_class!
loc:@discriminator/dense/bias*
_output_shapes
:?
?
1discriminator/dense/bias/Adam_1/Initializer/zerosConst*
valueB?*    *
dtype0*+
_class!
loc:@discriminator/dense/bias*
_output_shapes
:?
?
discriminator/dense/bias/Adam_1
VariableV2*
dtype0*
shared_name *
shape:?*
	container *+
_class!
loc:@discriminator/dense/bias*
_output_shapes
:?
?
&discriminator/dense/bias/Adam_1/AssignAssigndiscriminator/dense/bias/Adam_11discriminator/dense/bias/Adam_1/Initializer/zeros*
T0*
use_locking(*
validate_shape(*+
_class!
loc:@discriminator/dense/bias*
_output_shapes
:?
?
$discriminator/dense/bias/Adam_1/readIdentitydiscriminator/dense/bias/Adam_1*
T0*+
_class!
loc:@discriminator/dense/bias*
_output_shapes
:?
O

Adam/beta1Const*
valueB
 *   ?*
dtype0*
_output_shapes
: 
O

Adam/beta2Const*
valueB
 *w??*
dtype0*
_output_shapes
: 
Q
Adam/epsilonConst*
valueB
 *w?+2*
dtype0*
_output_shapes
: 
?
1Adam/update_discriminator/conv2d/kernel/ApplyAdam	ApplyAdamdiscriminator/conv2d/kernel discriminator/conv2d/kernel/Adam"discriminator/conv2d/kernel/Adam_1beta1_power/readbeta2_power/readVariable/read
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_19*
T0*
use_locking( *
use_nesterov( *.
_class$
" loc:@discriminator/conv2d/kernel*&
_output_shapes
:@
?
/Adam/update_discriminator/conv2d/bias/ApplyAdam	ApplyAdamdiscriminator/conv2d/biasdiscriminator/conv2d/bias/Adam discriminator/conv2d/bias/Adam_1beta1_power/readbeta2_power/readVariable/read
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_18*
T0*
use_locking( *
use_nesterov( *,
_class"
 loc:@discriminator/conv2d/bias*
_output_shapes
:@
?
3Adam/update_discriminator/conv2d_1/kernel/ApplyAdam	ApplyAdamdiscriminator/conv2d_1/kernel"discriminator/conv2d_1/kernel/Adam$discriminator/conv2d_1/kernel/Adam_1beta1_power/readbeta2_power/readVariable/read
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_17*
T0*
use_locking( *
use_nesterov( *0
_class&
$"loc:@discriminator/conv2d_1/kernel*&
_output_shapes
:@@
?
1Adam/update_discriminator/conv2d_1/bias/ApplyAdam	ApplyAdamdiscriminator/conv2d_1/bias discriminator/conv2d_1/bias/Adam"discriminator/conv2d_1/bias/Adam_1beta1_power/readbeta2_power/readVariable/read
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_16*
T0*
use_locking( *
use_nesterov( *.
_class$
" loc:@discriminator/conv2d_1/bias*
_output_shapes
:@
?
3Adam/update_discriminator/conv2d_2/kernel/ApplyAdam	ApplyAdamdiscriminator/conv2d_2/kernel"discriminator/conv2d_2/kernel/Adam$discriminator/conv2d_2/kernel/Adam_1beta1_power/readbeta2_power/readVariable/read
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_15*
T0*
use_locking( *
use_nesterov( *0
_class&
$"loc:@discriminator/conv2d_2/kernel*&
_output_shapes
:@@
?
1Adam/update_discriminator/conv2d_2/bias/ApplyAdam	ApplyAdamdiscriminator/conv2d_2/bias discriminator/conv2d_2/bias/Adam"discriminator/conv2d_2/bias/Adam_1beta1_power/readbeta2_power/readVariable/read
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_14*
T0*
use_locking( *
use_nesterov( *.
_class$
" loc:@discriminator/conv2d_2/bias*
_output_shapes
:@
?
3Adam/update_discriminator/conv2d_3/kernel/ApplyAdam	ApplyAdamdiscriminator/conv2d_3/kernel"discriminator/conv2d_3/kernel/Adam$discriminator/conv2d_3/kernel/Adam_1beta1_power/readbeta2_power/readVariable/read
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_13*
T0*
use_locking( *
use_nesterov( *0
_class&
$"loc:@discriminator/conv2d_3/kernel*'
_output_shapes
:@?
?
1Adam/update_discriminator/conv2d_3/bias/ApplyAdam	ApplyAdamdiscriminator/conv2d_3/bias discriminator/conv2d_3/bias/Adam"discriminator/conv2d_3/bias/Adam_1beta1_power/readbeta2_power/readVariable/read
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_12*
T0*
use_locking( *
use_nesterov( *.
_class$
" loc:@discriminator/conv2d_3/bias*
_output_shapes	
:?
?
3Adam/update_discriminator/conv2d_4/kernel/ApplyAdam	ApplyAdamdiscriminator/conv2d_4/kernel"discriminator/conv2d_4/kernel/Adam$discriminator/conv2d_4/kernel/Adam_1beta1_power/readbeta2_power/readVariable/read
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_11*
T0*
use_locking( *
use_nesterov( *0
_class&
$"loc:@discriminator/conv2d_4/kernel*(
_output_shapes
:??
?
1Adam/update_discriminator/conv2d_4/bias/ApplyAdam	ApplyAdamdiscriminator/conv2d_4/bias discriminator/conv2d_4/bias/Adam"discriminator/conv2d_4/bias/Adam_1beta1_power/readbeta2_power/readVariable/read
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_10*
T0*
use_locking( *
use_nesterov( *.
_class$
" loc:@discriminator/conv2d_4/bias*
_output_shapes	
:?
?
3Adam/update_discriminator/conv2d_5/kernel/ApplyAdam	ApplyAdamdiscriminator/conv2d_5/kernel"discriminator/conv2d_5/kernel/Adam$discriminator/conv2d_5/kernel/Adam_1beta1_power/readbeta2_power/readVariable/read
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_9*
T0*
use_locking( *
use_nesterov( *0
_class&
$"loc:@discriminator/conv2d_5/kernel*(
_output_shapes
:??
?
1Adam/update_discriminator/conv2d_5/bias/ApplyAdam	ApplyAdamdiscriminator/conv2d_5/bias discriminator/conv2d_5/bias/Adam"discriminator/conv2d_5/bias/Adam_1beta1_power/readbeta2_power/readVariable/read
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_8*
T0*
use_locking( *
use_nesterov( *.
_class$
" loc:@discriminator/conv2d_5/bias*
_output_shapes	
:?
?
0Adam/update_discriminator/dense/kernel/ApplyAdam	ApplyAdamdiscriminator/dense/kerneldiscriminator/dense/kernel/Adam!discriminator/dense/kernel/Adam_1beta1_power/readbeta2_power/readVariable/read
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_7*
T0*
use_locking( *
use_nesterov( *-
_class#
!loc:@discriminator/dense/kernel*
_output_shapes
:	??
?
.Adam/update_discriminator/dense/bias/ApplyAdam	ApplyAdamdiscriminator/dense/biasdiscriminator/dense/bias/Adamdiscriminator/dense/bias/Adam_1beta1_power/readbeta2_power/readVariable/read
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_6*
T0*
use_locking( *
use_nesterov( *+
_class!
loc:@discriminator/dense/bias*
_output_shapes
:?
?
Adam/mulMulbeta1_power/read
Adam/beta10^Adam/update_discriminator/conv2d/bias/ApplyAdam2^Adam/update_discriminator/conv2d/kernel/ApplyAdam2^Adam/update_discriminator/conv2d_1/bias/ApplyAdam4^Adam/update_discriminator/conv2d_1/kernel/ApplyAdam2^Adam/update_discriminator/conv2d_2/bias/ApplyAdam4^Adam/update_discriminator/conv2d_2/kernel/ApplyAdam2^Adam/update_discriminator/conv2d_3/bias/ApplyAdam4^Adam/update_discriminator/conv2d_3/kernel/ApplyAdam2^Adam/update_discriminator/conv2d_4/bias/ApplyAdam4^Adam/update_discriminator/conv2d_4/kernel/ApplyAdam2^Adam/update_discriminator/conv2d_5/bias/ApplyAdam4^Adam/update_discriminator/conv2d_5/kernel/ApplyAdam/^Adam/update_discriminator/dense/bias/ApplyAdam1^Adam/update_discriminator/dense/kernel/ApplyAdam*
T0*,
_class"
 loc:@discriminator/conv2d/bias*
_output_shapes
: 
?
Adam/AssignAssignbeta1_powerAdam/mul*
T0*
use_locking( *
validate_shape(*,
_class"
 loc:@discriminator/conv2d/bias*
_output_shapes
: 
?

Adam/mul_1Mulbeta2_power/read
Adam/beta20^Adam/update_discriminator/conv2d/bias/ApplyAdam2^Adam/update_discriminator/conv2d/kernel/ApplyAdam2^Adam/update_discriminator/conv2d_1/bias/ApplyAdam4^Adam/update_discriminator/conv2d_1/kernel/ApplyAdam2^Adam/update_discriminator/conv2d_2/bias/ApplyAdam4^Adam/update_discriminator/conv2d_2/kernel/ApplyAdam2^Adam/update_discriminator/conv2d_3/bias/ApplyAdam4^Adam/update_discriminator/conv2d_3/kernel/ApplyAdam2^Adam/update_discriminator/conv2d_4/bias/ApplyAdam4^Adam/update_discriminator/conv2d_4/kernel/ApplyAdam2^Adam/update_discriminator/conv2d_5/bias/ApplyAdam4^Adam/update_discriminator/conv2d_5/kernel/ApplyAdam/^Adam/update_discriminator/dense/bias/ApplyAdam1^Adam/update_discriminator/dense/kernel/ApplyAdam*
T0*,
_class"
 loc:@discriminator/conv2d/bias*
_output_shapes
: 
?
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
T0*
use_locking( *
validate_shape(*,
_class"
 loc:@discriminator/conv2d/bias*
_output_shapes
: 
?
AdamNoOp^Adam/Assign^Adam/Assign_10^Adam/update_discriminator/conv2d/bias/ApplyAdam2^Adam/update_discriminator/conv2d/kernel/ApplyAdam2^Adam/update_discriminator/conv2d_1/bias/ApplyAdam4^Adam/update_discriminator/conv2d_1/kernel/ApplyAdam2^Adam/update_discriminator/conv2d_2/bias/ApplyAdam4^Adam/update_discriminator/conv2d_2/kernel/ApplyAdam2^Adam/update_discriminator/conv2d_3/bias/ApplyAdam4^Adam/update_discriminator/conv2d_3/kernel/ApplyAdam2^Adam/update_discriminator/conv2d_4/bias/ApplyAdam4^Adam/update_discriminator/conv2d_4/kernel/ApplyAdam2^Adam/update_discriminator/conv2d_5/bias/ApplyAdam4^Adam/update_discriminator/conv2d_5/kernel/ApplyAdam/^Adam/update_discriminator/dense/bias/ApplyAdam1^Adam/update_discriminator/dense/kernel/ApplyAdam
T
gradients_1/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Z
gradients_1/grad_ys_0Const*
valueB
 *  ??*
dtype0*
_output_shapes
: 
u
gradients_1/FillFillgradients_1/Shapegradients_1/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
o
%gradients_1/Mean_4_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
?
gradients_1/Mean_4_grad/ReshapeReshapegradients_1/Fill%gradients_1/Mean_4_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
h
gradients_1/Mean_4_grad/ConstConst*
valueB:?*
dtype0*
_output_shapes
:
?
gradients_1/Mean_4_grad/TileTilegradients_1/Mean_4_grad/Reshapegradients_1/Mean_4_grad/Const*
T0*

Tmultiples0*
_output_shapes	
:?
d
gradients_1/Mean_4_grad/Const_1Const*
valueB
 *   C*
dtype0*
_output_shapes
: 
?
gradients_1/Mean_4_grad/truedivRealDivgradients_1/Mean_4_grad/Tilegradients_1/Mean_4_grad/Const_1*
T0*
_output_shapes	
:?
L
gradients_1/Abs_grad/SignSignsub*
T0*
_output_shapes	
:?
?
gradients_1/Abs_grad/mulMulgradients_1/Mean_4_grad/truedivgradients_1/Abs_grad/Sign*
T0*
_output_shapes	
:?
_
gradients_1/sub_grad/NegNeggradients_1/Abs_grad/mul*
T0*
_output_shapes	
:?
c
%gradients_1/sub_grad/tuple/group_depsNoOp^gradients_1/Abs_grad/mul^gradients_1/sub_grad/Neg
?
-gradients_1/sub_grad/tuple/control_dependencyIdentitygradients_1/Abs_grad/mul&^gradients_1/sub_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients_1/Abs_grad/mul*
_output_shapes	
:?
?
/gradients_1/sub_grad/tuple/control_dependency_1Identitygradients_1/sub_grad/Neg&^gradients_1/sub_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients_1/sub_grad/Neg*
_output_shapes	
:?
q
gradients_1/Mean_3_grad/ShapeShapediscriminator_1/Mean*
T0*
out_type0*
_output_shapes
:
?
gradients_1/Mean_3_grad/SizeConst*
value	B :*
dtype0*0
_class&
$"loc:@gradients_1/Mean_3_grad/Shape*
_output_shapes
: 
?
gradients_1/Mean_3_grad/addAddMean_3/reduction_indicesgradients_1/Mean_3_grad/Size*
T0*0
_class&
$"loc:@gradients_1/Mean_3_grad/Shape*
_output_shapes
: 
?
gradients_1/Mean_3_grad/modFloorModgradients_1/Mean_3_grad/addgradients_1/Mean_3_grad/Size*
T0*0
_class&
$"loc:@gradients_1/Mean_3_grad/Shape*
_output_shapes
: 
?
gradients_1/Mean_3_grad/Shape_1Const*
valueB *
dtype0*0
_class&
$"loc:@gradients_1/Mean_3_grad/Shape*
_output_shapes
: 
?
#gradients_1/Mean_3_grad/range/startConst*
value	B : *
dtype0*0
_class&
$"loc:@gradients_1/Mean_3_grad/Shape*
_output_shapes
: 
?
#gradients_1/Mean_3_grad/range/deltaConst*
value	B :*
dtype0*0
_class&
$"loc:@gradients_1/Mean_3_grad/Shape*
_output_shapes
: 
?
gradients_1/Mean_3_grad/rangeRange#gradients_1/Mean_3_grad/range/startgradients_1/Mean_3_grad/Size#gradients_1/Mean_3_grad/range/delta*

Tidx0*0
_class&
$"loc:@gradients_1/Mean_3_grad/Shape*
_output_shapes
:
?
"gradients_1/Mean_3_grad/Fill/valueConst*
value	B :*
dtype0*0
_class&
$"loc:@gradients_1/Mean_3_grad/Shape*
_output_shapes
: 
?
gradients_1/Mean_3_grad/FillFillgradients_1/Mean_3_grad/Shape_1"gradients_1/Mean_3_grad/Fill/value*
T0*

index_type0*0
_class&
$"loc:@gradients_1/Mean_3_grad/Shape*
_output_shapes
: 
?
%gradients_1/Mean_3_grad/DynamicStitchDynamicStitchgradients_1/Mean_3_grad/rangegradients_1/Mean_3_grad/modgradients_1/Mean_3_grad/Shapegradients_1/Mean_3_grad/Fill*
T0*
N*0
_class&
$"loc:@gradients_1/Mean_3_grad/Shape*
_output_shapes
:
?
!gradients_1/Mean_3_grad/Maximum/yConst*
value	B :*
dtype0*0
_class&
$"loc:@gradients_1/Mean_3_grad/Shape*
_output_shapes
: 
?
gradients_1/Mean_3_grad/MaximumMaximum%gradients_1/Mean_3_grad/DynamicStitch!gradients_1/Mean_3_grad/Maximum/y*
T0*0
_class&
$"loc:@gradients_1/Mean_3_grad/Shape*
_output_shapes
:
?
 gradients_1/Mean_3_grad/floordivFloorDivgradients_1/Mean_3_grad/Shapegradients_1/Mean_3_grad/Maximum*
T0*0
_class&
$"loc:@gradients_1/Mean_3_grad/Shape*
_output_shapes
:
?
gradients_1/Mean_3_grad/ReshapeReshape/gradients_1/sub_grad/tuple/control_dependency_1%gradients_1/Mean_3_grad/DynamicStitch*
T0*
Tshape0*0
_output_shapes
:??????????????????
?
gradients_1/Mean_3_grad/TileTilegradients_1/Mean_3_grad/Reshape gradients_1/Mean_3_grad/floordiv*
T0*

Tmultiples0*0
_output_shapes
:??????????????????
s
gradients_1/Mean_3_grad/Shape_2Shapediscriminator_1/Mean*
T0*
out_type0*
_output_shapes
:
j
gradients_1/Mean_3_grad/Shape_3Const*
valueB:?*
dtype0*
_output_shapes
:
g
gradients_1/Mean_3_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
?
gradients_1/Mean_3_grad/ProdProdgradients_1/Mean_3_grad/Shape_2gradients_1/Mean_3_grad/Const*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
i
gradients_1/Mean_3_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
?
gradients_1/Mean_3_grad/Prod_1Prodgradients_1/Mean_3_grad/Shape_3gradients_1/Mean_3_grad/Const_1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
e
#gradients_1/Mean_3_grad/Maximum_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
?
!gradients_1/Mean_3_grad/Maximum_1Maximumgradients_1/Mean_3_grad/Prod_1#gradients_1/Mean_3_grad/Maximum_1/y*
T0*
_output_shapes
: 
?
"gradients_1/Mean_3_grad/floordiv_1FloorDivgradients_1/Mean_3_grad/Prod!gradients_1/Mean_3_grad/Maximum_1*
T0*
_output_shapes
: 
?
gradients_1/Mean_3_grad/CastCast"gradients_1/Mean_3_grad/floordiv_1*

DstT0*
Truncate( *

SrcT0*
_output_shapes
: 
?
gradients_1/Mean_3_grad/truedivRealDivgradients_1/Mean_3_grad/Tilegradients_1/Mean_3_grad/Cast*
T0*(
_output_shapes
:??????????
?
+gradients_1/discriminator_1/Mean_grad/ShapeShapediscriminator_1/conv2d_5/Relu*
T0*
out_type0*
_output_shapes
:
?
*gradients_1/discriminator_1/Mean_grad/SizeConst*
value	B :*
dtype0*>
_class4
20loc:@gradients_1/discriminator_1/Mean_grad/Shape*
_output_shapes
: 
?
)gradients_1/discriminator_1/Mean_grad/addAdd&discriminator_1/Mean/reduction_indices*gradients_1/discriminator_1/Mean_grad/Size*
T0*>
_class4
20loc:@gradients_1/discriminator_1/Mean_grad/Shape*
_output_shapes
:
?
)gradients_1/discriminator_1/Mean_grad/modFloorMod)gradients_1/discriminator_1/Mean_grad/add*gradients_1/discriminator_1/Mean_grad/Size*
T0*>
_class4
20loc:@gradients_1/discriminator_1/Mean_grad/Shape*
_output_shapes
:
?
-gradients_1/discriminator_1/Mean_grad/Shape_1Const*
valueB:*
dtype0*>
_class4
20loc:@gradients_1/discriminator_1/Mean_grad/Shape*
_output_shapes
:
?
1gradients_1/discriminator_1/Mean_grad/range/startConst*
value	B : *
dtype0*>
_class4
20loc:@gradients_1/discriminator_1/Mean_grad/Shape*
_output_shapes
: 
?
1gradients_1/discriminator_1/Mean_grad/range/deltaConst*
value	B :*
dtype0*>
_class4
20loc:@gradients_1/discriminator_1/Mean_grad/Shape*
_output_shapes
: 
?
+gradients_1/discriminator_1/Mean_grad/rangeRange1gradients_1/discriminator_1/Mean_grad/range/start*gradients_1/discriminator_1/Mean_grad/Size1gradients_1/discriminator_1/Mean_grad/range/delta*

Tidx0*>
_class4
20loc:@gradients_1/discriminator_1/Mean_grad/Shape*
_output_shapes
:
?
0gradients_1/discriminator_1/Mean_grad/Fill/valueConst*
value	B :*
dtype0*>
_class4
20loc:@gradients_1/discriminator_1/Mean_grad/Shape*
_output_shapes
: 
?
*gradients_1/discriminator_1/Mean_grad/FillFill-gradients_1/discriminator_1/Mean_grad/Shape_10gradients_1/discriminator_1/Mean_grad/Fill/value*
T0*

index_type0*>
_class4
20loc:@gradients_1/discriminator_1/Mean_grad/Shape*
_output_shapes
:
?
3gradients_1/discriminator_1/Mean_grad/DynamicStitchDynamicStitch+gradients_1/discriminator_1/Mean_grad/range)gradients_1/discriminator_1/Mean_grad/mod+gradients_1/discriminator_1/Mean_grad/Shape*gradients_1/discriminator_1/Mean_grad/Fill*
T0*
N*>
_class4
20loc:@gradients_1/discriminator_1/Mean_grad/Shape*
_output_shapes
:
?
/gradients_1/discriminator_1/Mean_grad/Maximum/yConst*
value	B :*
dtype0*>
_class4
20loc:@gradients_1/discriminator_1/Mean_grad/Shape*
_output_shapes
: 
?
-gradients_1/discriminator_1/Mean_grad/MaximumMaximum3gradients_1/discriminator_1/Mean_grad/DynamicStitch/gradients_1/discriminator_1/Mean_grad/Maximum/y*
T0*>
_class4
20loc:@gradients_1/discriminator_1/Mean_grad/Shape*
_output_shapes
:
?
.gradients_1/discriminator_1/Mean_grad/floordivFloorDiv+gradients_1/discriminator_1/Mean_grad/Shape-gradients_1/discriminator_1/Mean_grad/Maximum*
T0*>
_class4
20loc:@gradients_1/discriminator_1/Mean_grad/Shape*
_output_shapes
:
?
-gradients_1/discriminator_1/Mean_grad/ReshapeReshapegradients_1/Mean_3_grad/truediv3gradients_1/discriminator_1/Mean_grad/DynamicStitch*
T0*
Tshape0*J
_output_shapes8
6:4????????????????????????????????????
?
*gradients_1/discriminator_1/Mean_grad/TileTile-gradients_1/discriminator_1/Mean_grad/Reshape.gradients_1/discriminator_1/Mean_grad/floordiv*
T0*

Tmultiples0*J
_output_shapes8
6:4????????????????????????????????????
?
-gradients_1/discriminator_1/Mean_grad/Shape_2Shapediscriminator_1/conv2d_5/Relu*
T0*
out_type0*
_output_shapes
:
?
-gradients_1/discriminator_1/Mean_grad/Shape_3Shapediscriminator_1/Mean*
T0*
out_type0*
_output_shapes
:
u
+gradients_1/discriminator_1/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
?
*gradients_1/discriminator_1/Mean_grad/ProdProd-gradients_1/discriminator_1/Mean_grad/Shape_2+gradients_1/discriminator_1/Mean_grad/Const*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
w
-gradients_1/discriminator_1/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
?
,gradients_1/discriminator_1/Mean_grad/Prod_1Prod-gradients_1/discriminator_1/Mean_grad/Shape_3-gradients_1/discriminator_1/Mean_grad/Const_1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
s
1gradients_1/discriminator_1/Mean_grad/Maximum_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
?
/gradients_1/discriminator_1/Mean_grad/Maximum_1Maximum,gradients_1/discriminator_1/Mean_grad/Prod_11gradients_1/discriminator_1/Mean_grad/Maximum_1/y*
T0*
_output_shapes
: 
?
0gradients_1/discriminator_1/Mean_grad/floordiv_1FloorDiv*gradients_1/discriminator_1/Mean_grad/Prod/gradients_1/discriminator_1/Mean_grad/Maximum_1*
T0*
_output_shapes
: 
?
*gradients_1/discriminator_1/Mean_grad/CastCast0gradients_1/discriminator_1/Mean_grad/floordiv_1*

DstT0*
Truncate( *

SrcT0*
_output_shapes
: 
?
-gradients_1/discriminator_1/Mean_grad/truedivRealDiv*gradients_1/discriminator_1/Mean_grad/Tile*gradients_1/discriminator_1/Mean_grad/Cast*
T0*0
_output_shapes
:??????????
?
7gradients_1/discriminator_1/conv2d_5/Relu_grad/ReluGradReluGrad-gradients_1/discriminator_1/Mean_grad/truedivdiscriminator_1/conv2d_5/Relu*
T0*0
_output_shapes
:??????????
?
=gradients_1/discriminator_1/conv2d_5/BiasAdd_grad/BiasAddGradBiasAddGrad7gradients_1/discriminator_1/conv2d_5/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:?
?
Bgradients_1/discriminator_1/conv2d_5/BiasAdd_grad/tuple/group_depsNoOp>^gradients_1/discriminator_1/conv2d_5/BiasAdd_grad/BiasAddGrad8^gradients_1/discriminator_1/conv2d_5/Relu_grad/ReluGrad
?
Jgradients_1/discriminator_1/conv2d_5/BiasAdd_grad/tuple/control_dependencyIdentity7gradients_1/discriminator_1/conv2d_5/Relu_grad/ReluGradC^gradients_1/discriminator_1/conv2d_5/BiasAdd_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients_1/discriminator_1/conv2d_5/Relu_grad/ReluGrad*0
_output_shapes
:??????????
?
Lgradients_1/discriminator_1/conv2d_5/BiasAdd_grad/tuple/control_dependency_1Identity=gradients_1/discriminator_1/conv2d_5/BiasAdd_grad/BiasAddGradC^gradients_1/discriminator_1/conv2d_5/BiasAdd_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients_1/discriminator_1/conv2d_5/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:?
?
7gradients_1/discriminator_1/conv2d_5/Conv2D_grad/ShapeNShapeNdiscriminator_1/Relu_2"discriminator/conv2d_5/kernel/read*
T0*
out_type0*
N* 
_output_shapes
::
?
Dgradients_1/discriminator_1/conv2d_5/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput7gradients_1/discriminator_1/conv2d_5/Conv2D_grad/ShapeN"discriminator/conv2d_5/kernel/readJgradients_1/discriminator_1/conv2d_5/BiasAdd_grad/tuple/control_dependency*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingVALID*
use_cudnn_on_gpu(*0
_output_shapes
:??????????
?
Egradients_1/discriminator_1/conv2d_5/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterdiscriminator_1/Relu_29gradients_1/discriminator_1/conv2d_5/Conv2D_grad/ShapeN:1Jgradients_1/discriminator_1/conv2d_5/BiasAdd_grad/tuple/control_dependency*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingVALID*
use_cudnn_on_gpu(*(
_output_shapes
:??
?
Agradients_1/discriminator_1/conv2d_5/Conv2D_grad/tuple/group_depsNoOpF^gradients_1/discriminator_1/conv2d_5/Conv2D_grad/Conv2DBackpropFilterE^gradients_1/discriminator_1/conv2d_5/Conv2D_grad/Conv2DBackpropInput
?
Igradients_1/discriminator_1/conv2d_5/Conv2D_grad/tuple/control_dependencyIdentityDgradients_1/discriminator_1/conv2d_5/Conv2D_grad/Conv2DBackpropInputB^gradients_1/discriminator_1/conv2d_5/Conv2D_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_1/discriminator_1/conv2d_5/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:??????????
?
Kgradients_1/discriminator_1/conv2d_5/Conv2D_grad/tuple/control_dependency_1IdentityEgradients_1/discriminator_1/conv2d_5/Conv2D_grad/Conv2DBackpropFilterB^gradients_1/discriminator_1/conv2d_5/Conv2D_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients_1/discriminator_1/conv2d_5/Conv2D_grad/Conv2DBackpropFilter*(
_output_shapes
:??
?
0gradients_1/discriminator_1/Relu_2_grad/ReluGradReluGradIgradients_1/discriminator_1/conv2d_5/Conv2D_grad/tuple/control_dependencydiscriminator_1/Relu_2*
T0*0
_output_shapes
:??????????
?
gradients_1/zeros_like	ZerosLike6discriminator_1/batch_normalization_2/FusedBatchNorm:1*
T0*
_output_shapes	
:?
?
gradients_1/zeros_like_1	ZerosLike6discriminator_1/batch_normalization_2/FusedBatchNorm:2*
T0*
_output_shapes	
:?
?
gradients_1/zeros_like_2	ZerosLike6discriminator_1/batch_normalization_2/FusedBatchNorm:3*
T0*
_output_shapes	
:?
?
gradients_1/zeros_like_3	ZerosLike6discriminator_1/batch_normalization_2/FusedBatchNorm:4*
T0*
_output_shapes	
:?
?
Xgradients_1/discriminator_1/batch_normalization_2/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGrad0gradients_1/discriminator_1/Relu_2_grad/ReluGrad discriminator_1/conv2d_4/BiasAdd.discriminator/batch_normalization_2/gamma/read4discriminator/batch_normalization_2/moving_mean/read8discriminator/batch_normalization_2/moving_variance/read*
is_training( *
T0*
data_formatNHWC*
epsilon%o?:*L
_output_shapes:
8:??????????:?:?:?:?
?
Vgradients_1/discriminator_1/batch_normalization_2/FusedBatchNorm_grad/tuple/group_depsNoOpY^gradients_1/discriminator_1/batch_normalization_2/FusedBatchNorm_grad/FusedBatchNormGrad
?
^gradients_1/discriminator_1/batch_normalization_2/FusedBatchNorm_grad/tuple/control_dependencyIdentityXgradients_1/discriminator_1/batch_normalization_2/FusedBatchNorm_grad/FusedBatchNormGradW^gradients_1/discriminator_1/batch_normalization_2/FusedBatchNorm_grad/tuple/group_deps*
T0*k
_classa
_]loc:@gradients_1/discriminator_1/batch_normalization_2/FusedBatchNorm_grad/FusedBatchNormGrad*0
_output_shapes
:??????????
?
`gradients_1/discriminator_1/batch_normalization_2/FusedBatchNorm_grad/tuple/control_dependency_1IdentityZgradients_1/discriminator_1/batch_normalization_2/FusedBatchNorm_grad/FusedBatchNormGrad:1W^gradients_1/discriminator_1/batch_normalization_2/FusedBatchNorm_grad/tuple/group_deps*
T0*k
_classa
_]loc:@gradients_1/discriminator_1/batch_normalization_2/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes	
:?
?
`gradients_1/discriminator_1/batch_normalization_2/FusedBatchNorm_grad/tuple/control_dependency_2IdentityZgradients_1/discriminator_1/batch_normalization_2/FusedBatchNorm_grad/FusedBatchNormGrad:2W^gradients_1/discriminator_1/batch_normalization_2/FusedBatchNorm_grad/tuple/group_deps*
T0*k
_classa
_]loc:@gradients_1/discriminator_1/batch_normalization_2/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes	
:?
?
=gradients_1/discriminator_1/conv2d_4/BiasAdd_grad/BiasAddGradBiasAddGrad^gradients_1/discriminator_1/batch_normalization_2/FusedBatchNorm_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes	
:?
?
Bgradients_1/discriminator_1/conv2d_4/BiasAdd_grad/tuple/group_depsNoOp_^gradients_1/discriminator_1/batch_normalization_2/FusedBatchNorm_grad/tuple/control_dependency>^gradients_1/discriminator_1/conv2d_4/BiasAdd_grad/BiasAddGrad
?
Jgradients_1/discriminator_1/conv2d_4/BiasAdd_grad/tuple/control_dependencyIdentity^gradients_1/discriminator_1/batch_normalization_2/FusedBatchNorm_grad/tuple/control_dependencyC^gradients_1/discriminator_1/conv2d_4/BiasAdd_grad/tuple/group_deps*
T0*k
_classa
_]loc:@gradients_1/discriminator_1/batch_normalization_2/FusedBatchNorm_grad/FusedBatchNormGrad*0
_output_shapes
:??????????
?
Lgradients_1/discriminator_1/conv2d_4/BiasAdd_grad/tuple/control_dependency_1Identity=gradients_1/discriminator_1/conv2d_4/BiasAdd_grad/BiasAddGradC^gradients_1/discriminator_1/conv2d_4/BiasAdd_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients_1/discriminator_1/conv2d_4/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:?
?
7gradients_1/discriminator_1/conv2d_4/Conv2D_grad/ShapeNShapeNdiscriminator_1/Relu_1"discriminator/conv2d_4/kernel/read*
T0*
out_type0*
N* 
_output_shapes
::
?
Dgradients_1/discriminator_1/conv2d_4/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput7gradients_1/discriminator_1/conv2d_4/Conv2D_grad/ShapeN"discriminator/conv2d_4/kernel/readJgradients_1/discriminator_1/conv2d_4/BiasAdd_grad/tuple/control_dependency*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(*0
_output_shapes
:??????????
?
Egradients_1/discriminator_1/conv2d_4/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterdiscriminator_1/Relu_19gradients_1/discriminator_1/conv2d_4/Conv2D_grad/ShapeN:1Jgradients_1/discriminator_1/conv2d_4/BiasAdd_grad/tuple/control_dependency*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(*(
_output_shapes
:??
?
Agradients_1/discriminator_1/conv2d_4/Conv2D_grad/tuple/group_depsNoOpF^gradients_1/discriminator_1/conv2d_4/Conv2D_grad/Conv2DBackpropFilterE^gradients_1/discriminator_1/conv2d_4/Conv2D_grad/Conv2DBackpropInput
?
Igradients_1/discriminator_1/conv2d_4/Conv2D_grad/tuple/control_dependencyIdentityDgradients_1/discriminator_1/conv2d_4/Conv2D_grad/Conv2DBackpropInputB^gradients_1/discriminator_1/conv2d_4/Conv2D_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_1/discriminator_1/conv2d_4/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:??????????
?
Kgradients_1/discriminator_1/conv2d_4/Conv2D_grad/tuple/control_dependency_1IdentityEgradients_1/discriminator_1/conv2d_4/Conv2D_grad/Conv2DBackpropFilterB^gradients_1/discriminator_1/conv2d_4/Conv2D_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients_1/discriminator_1/conv2d_4/Conv2D_grad/Conv2DBackpropFilter*(
_output_shapes
:??
?
0gradients_1/discriminator_1/Relu_1_grad/ReluGradReluGradIgradients_1/discriminator_1/conv2d_4/Conv2D_grad/tuple/control_dependencydiscriminator_1/Relu_1*
T0*0
_output_shapes
:??????????
?
gradients_1/zeros_like_4	ZerosLike6discriminator_1/batch_normalization_1/FusedBatchNorm:1*
T0*
_output_shapes	
:?
?
gradients_1/zeros_like_5	ZerosLike6discriminator_1/batch_normalization_1/FusedBatchNorm:2*
T0*
_output_shapes	
:?
?
gradients_1/zeros_like_6	ZerosLike6discriminator_1/batch_normalization_1/FusedBatchNorm:3*
T0*
_output_shapes	
:?
?
gradients_1/zeros_like_7	ZerosLike6discriminator_1/batch_normalization_1/FusedBatchNorm:4*
T0*
_output_shapes	
:?
?
Xgradients_1/discriminator_1/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGrad0gradients_1/discriminator_1/Relu_1_grad/ReluGrad discriminator_1/conv2d_3/BiasAdd.discriminator/batch_normalization_1/gamma/read4discriminator/batch_normalization_1/moving_mean/read8discriminator/batch_normalization_1/moving_variance/read*
is_training( *
T0*
data_formatNHWC*
epsilon%o?:*L
_output_shapes:
8:??????????:?:?:?:?
?
Vgradients_1/discriminator_1/batch_normalization_1/FusedBatchNorm_grad/tuple/group_depsNoOpY^gradients_1/discriminator_1/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGrad
?
^gradients_1/discriminator_1/batch_normalization_1/FusedBatchNorm_grad/tuple/control_dependencyIdentityXgradients_1/discriminator_1/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGradW^gradients_1/discriminator_1/batch_normalization_1/FusedBatchNorm_grad/tuple/group_deps*
T0*k
_classa
_]loc:@gradients_1/discriminator_1/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGrad*0
_output_shapes
:??????????
?
`gradients_1/discriminator_1/batch_normalization_1/FusedBatchNorm_grad/tuple/control_dependency_1IdentityZgradients_1/discriminator_1/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGrad:1W^gradients_1/discriminator_1/batch_normalization_1/FusedBatchNorm_grad/tuple/group_deps*
T0*k
_classa
_]loc:@gradients_1/discriminator_1/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes	
:?
?
`gradients_1/discriminator_1/batch_normalization_1/FusedBatchNorm_grad/tuple/control_dependency_2IdentityZgradients_1/discriminator_1/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGrad:2W^gradients_1/discriminator_1/batch_normalization_1/FusedBatchNorm_grad/tuple/group_deps*
T0*k
_classa
_]loc:@gradients_1/discriminator_1/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes	
:?
?
=gradients_1/discriminator_1/conv2d_3/BiasAdd_grad/BiasAddGradBiasAddGrad^gradients_1/discriminator_1/batch_normalization_1/FusedBatchNorm_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes	
:?
?
Bgradients_1/discriminator_1/conv2d_3/BiasAdd_grad/tuple/group_depsNoOp_^gradients_1/discriminator_1/batch_normalization_1/FusedBatchNorm_grad/tuple/control_dependency>^gradients_1/discriminator_1/conv2d_3/BiasAdd_grad/BiasAddGrad
?
Jgradients_1/discriminator_1/conv2d_3/BiasAdd_grad/tuple/control_dependencyIdentity^gradients_1/discriminator_1/batch_normalization_1/FusedBatchNorm_grad/tuple/control_dependencyC^gradients_1/discriminator_1/conv2d_3/BiasAdd_grad/tuple/group_deps*
T0*k
_classa
_]loc:@gradients_1/discriminator_1/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGrad*0
_output_shapes
:??????????
?
Lgradients_1/discriminator_1/conv2d_3/BiasAdd_grad/tuple/control_dependency_1Identity=gradients_1/discriminator_1/conv2d_3/BiasAdd_grad/BiasAddGradC^gradients_1/discriminator_1/conv2d_3/BiasAdd_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients_1/discriminator_1/conv2d_3/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:?
?
7gradients_1/discriminator_1/conv2d_3/Conv2D_grad/ShapeNShapeN"discriminator_1/dropout_2/Identity"discriminator/conv2d_3/kernel/read*
T0*
out_type0*
N* 
_output_shapes
::
?
Dgradients_1/discriminator_1/conv2d_3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput7gradients_1/discriminator_1/conv2d_3/Conv2D_grad/ShapeN"discriminator/conv2d_3/kernel/readJgradients_1/discriminator_1/conv2d_3/BiasAdd_grad/tuple/control_dependency*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(*/
_output_shapes
:?????????@
?
Egradients_1/discriminator_1/conv2d_3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter"discriminator_1/dropout_2/Identity9gradients_1/discriminator_1/conv2d_3/Conv2D_grad/ShapeN:1Jgradients_1/discriminator_1/conv2d_3/BiasAdd_grad/tuple/control_dependency*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(*'
_output_shapes
:@?
?
Agradients_1/discriminator_1/conv2d_3/Conv2D_grad/tuple/group_depsNoOpF^gradients_1/discriminator_1/conv2d_3/Conv2D_grad/Conv2DBackpropFilterE^gradients_1/discriminator_1/conv2d_3/Conv2D_grad/Conv2DBackpropInput
?
Igradients_1/discriminator_1/conv2d_3/Conv2D_grad/tuple/control_dependencyIdentityDgradients_1/discriminator_1/conv2d_3/Conv2D_grad/Conv2DBackpropInputB^gradients_1/discriminator_1/conv2d_3/Conv2D_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_1/discriminator_1/conv2d_3/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:?????????@
?
Kgradients_1/discriminator_1/conv2d_3/Conv2D_grad/tuple/control_dependency_1IdentityEgradients_1/discriminator_1/conv2d_3/Conv2D_grad/Conv2DBackpropFilterB^gradients_1/discriminator_1/conv2d_3/Conv2D_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients_1/discriminator_1/conv2d_3/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:@?
?
.gradients_1/discriminator_1/Relu_grad/ReluGradReluGradIgradients_1/discriminator_1/conv2d_3/Conv2D_grad/tuple/control_dependencydiscriminator_1/Relu*
T0*/
_output_shapes
:?????????@
?
gradients_1/zeros_like_8	ZerosLike4discriminator_1/batch_normalization/FusedBatchNorm:1*
T0*
_output_shapes
:@
?
gradients_1/zeros_like_9	ZerosLike4discriminator_1/batch_normalization/FusedBatchNorm:2*
T0*
_output_shapes
:@
?
gradients_1/zeros_like_10	ZerosLike4discriminator_1/batch_normalization/FusedBatchNorm:3*
T0*
_output_shapes
:@
?
gradients_1/zeros_like_11	ZerosLike4discriminator_1/batch_normalization/FusedBatchNorm:4*
T0*
_output_shapes
:@
?
Vgradients_1/discriminator_1/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGrad.gradients_1/discriminator_1/Relu_grad/ReluGrad discriminator_1/conv2d_2/BiasAdd,discriminator/batch_normalization/gamma/read2discriminator/batch_normalization/moving_mean/read6discriminator/batch_normalization/moving_variance/read*
is_training( *
T0*
data_formatNHWC*
epsilon%o?:*G
_output_shapes5
3:?????????@:@:@:@:@
?
Tgradients_1/discriminator_1/batch_normalization/FusedBatchNorm_grad/tuple/group_depsNoOpW^gradients_1/discriminator_1/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGrad
?
\gradients_1/discriminator_1/batch_normalization/FusedBatchNorm_grad/tuple/control_dependencyIdentityVgradients_1/discriminator_1/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGradU^gradients_1/discriminator_1/batch_normalization/FusedBatchNorm_grad/tuple/group_deps*
T0*i
_class_
][loc:@gradients_1/discriminator_1/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGrad*/
_output_shapes
:?????????@
?
^gradients_1/discriminator_1/batch_normalization/FusedBatchNorm_grad/tuple/control_dependency_1IdentityXgradients_1/discriminator_1/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGrad:1U^gradients_1/discriminator_1/batch_normalization/FusedBatchNorm_grad/tuple/group_deps*
T0*i
_class_
][loc:@gradients_1/discriminator_1/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
:@
?
^gradients_1/discriminator_1/batch_normalization/FusedBatchNorm_grad/tuple/control_dependency_2IdentityXgradients_1/discriminator_1/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGrad:2U^gradients_1/discriminator_1/batch_normalization/FusedBatchNorm_grad/tuple/group_deps*
T0*i
_class_
][loc:@gradients_1/discriminator_1/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
:@
?
=gradients_1/discriminator_1/conv2d_2/BiasAdd_grad/BiasAddGradBiasAddGrad\gradients_1/discriminator_1/batch_normalization/FusedBatchNorm_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes
:@
?
Bgradients_1/discriminator_1/conv2d_2/BiasAdd_grad/tuple/group_depsNoOp]^gradients_1/discriminator_1/batch_normalization/FusedBatchNorm_grad/tuple/control_dependency>^gradients_1/discriminator_1/conv2d_2/BiasAdd_grad/BiasAddGrad
?
Jgradients_1/discriminator_1/conv2d_2/BiasAdd_grad/tuple/control_dependencyIdentity\gradients_1/discriminator_1/batch_normalization/FusedBatchNorm_grad/tuple/control_dependencyC^gradients_1/discriminator_1/conv2d_2/BiasAdd_grad/tuple/group_deps*
T0*i
_class_
][loc:@gradients_1/discriminator_1/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGrad*/
_output_shapes
:?????????@
?
Lgradients_1/discriminator_1/conv2d_2/BiasAdd_grad/tuple/control_dependency_1Identity=gradients_1/discriminator_1/conv2d_2/BiasAdd_grad/BiasAddGradC^gradients_1/discriminator_1/conv2d_2/BiasAdd_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients_1/discriminator_1/conv2d_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
?
7gradients_1/discriminator_1/conv2d_2/Conv2D_grad/ShapeNShapeNdiscriminator_1/conv2d_1/Relu"discriminator/conv2d_2/kernel/read*
T0*
out_type0*
N* 
_output_shapes
::
?
Dgradients_1/discriminator_1/conv2d_2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput7gradients_1/discriminator_1/conv2d_2/Conv2D_grad/ShapeN"discriminator/conv2d_2/kernel/readJgradients_1/discriminator_1/conv2d_2/BiasAdd_grad/tuple/control_dependency*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(*/
_output_shapes
:?????????@
?
Egradients_1/discriminator_1/conv2d_2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterdiscriminator_1/conv2d_1/Relu9gradients_1/discriminator_1/conv2d_2/Conv2D_grad/ShapeN:1Jgradients_1/discriminator_1/conv2d_2/BiasAdd_grad/tuple/control_dependency*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(*&
_output_shapes
:@@
?
Agradients_1/discriminator_1/conv2d_2/Conv2D_grad/tuple/group_depsNoOpF^gradients_1/discriminator_1/conv2d_2/Conv2D_grad/Conv2DBackpropFilterE^gradients_1/discriminator_1/conv2d_2/Conv2D_grad/Conv2DBackpropInput
?
Igradients_1/discriminator_1/conv2d_2/Conv2D_grad/tuple/control_dependencyIdentityDgradients_1/discriminator_1/conv2d_2/Conv2D_grad/Conv2DBackpropInputB^gradients_1/discriminator_1/conv2d_2/Conv2D_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_1/discriminator_1/conv2d_2/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:?????????@
?
Kgradients_1/discriminator_1/conv2d_2/Conv2D_grad/tuple/control_dependency_1IdentityEgradients_1/discriminator_1/conv2d_2/Conv2D_grad/Conv2DBackpropFilterB^gradients_1/discriminator_1/conv2d_2/Conv2D_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients_1/discriminator_1/conv2d_2/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:@@
?
7gradients_1/discriminator_1/conv2d_1/Relu_grad/ReluGradReluGradIgradients_1/discriminator_1/conv2d_2/Conv2D_grad/tuple/control_dependencydiscriminator_1/conv2d_1/Relu*
T0*/
_output_shapes
:?????????@
?
=gradients_1/discriminator_1/conv2d_1/BiasAdd_grad/BiasAddGradBiasAddGrad7gradients_1/discriminator_1/conv2d_1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
?
Bgradients_1/discriminator_1/conv2d_1/BiasAdd_grad/tuple/group_depsNoOp>^gradients_1/discriminator_1/conv2d_1/BiasAdd_grad/BiasAddGrad8^gradients_1/discriminator_1/conv2d_1/Relu_grad/ReluGrad
?
Jgradients_1/discriminator_1/conv2d_1/BiasAdd_grad/tuple/control_dependencyIdentity7gradients_1/discriminator_1/conv2d_1/Relu_grad/ReluGradC^gradients_1/discriminator_1/conv2d_1/BiasAdd_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients_1/discriminator_1/conv2d_1/Relu_grad/ReluGrad*/
_output_shapes
:?????????@
?
Lgradients_1/discriminator_1/conv2d_1/BiasAdd_grad/tuple/control_dependency_1Identity=gradients_1/discriminator_1/conv2d_1/BiasAdd_grad/BiasAddGradC^gradients_1/discriminator_1/conv2d_1/BiasAdd_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients_1/discriminator_1/conv2d_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
?
7gradients_1/discriminator_1/conv2d_1/Conv2D_grad/ShapeNShapeN"discriminator_1/dropout_1/Identity"discriminator/conv2d_1/kernel/read*
T0*
out_type0*
N* 
_output_shapes
::
?
Dgradients_1/discriminator_1/conv2d_1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput7gradients_1/discriminator_1/conv2d_1/Conv2D_grad/ShapeN"discriminator/conv2d_1/kernel/readJgradients_1/discriminator_1/conv2d_1/BiasAdd_grad/tuple/control_dependency*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(*/
_output_shapes
:?????????@
?
Egradients_1/discriminator_1/conv2d_1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter"discriminator_1/dropout_1/Identity9gradients_1/discriminator_1/conv2d_1/Conv2D_grad/ShapeN:1Jgradients_1/discriminator_1/conv2d_1/BiasAdd_grad/tuple/control_dependency*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(*&
_output_shapes
:@@
?
Agradients_1/discriminator_1/conv2d_1/Conv2D_grad/tuple/group_depsNoOpF^gradients_1/discriminator_1/conv2d_1/Conv2D_grad/Conv2DBackpropFilterE^gradients_1/discriminator_1/conv2d_1/Conv2D_grad/Conv2DBackpropInput
?
Igradients_1/discriminator_1/conv2d_1/Conv2D_grad/tuple/control_dependencyIdentityDgradients_1/discriminator_1/conv2d_1/Conv2D_grad/Conv2DBackpropInputB^gradients_1/discriminator_1/conv2d_1/Conv2D_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_1/discriminator_1/conv2d_1/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:?????????@
?
Kgradients_1/discriminator_1/conv2d_1/Conv2D_grad/tuple/control_dependency_1IdentityEgradients_1/discriminator_1/conv2d_1/Conv2D_grad/Conv2DBackpropFilterB^gradients_1/discriminator_1/conv2d_1/Conv2D_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients_1/discriminator_1/conv2d_1/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:@@
?
5gradients_1/discriminator_1/conv2d/Relu_grad/ReluGradReluGradIgradients_1/discriminator_1/conv2d_1/Conv2D_grad/tuple/control_dependencydiscriminator_1/conv2d/Relu*
T0*/
_output_shapes
:?????????@
?
;gradients_1/discriminator_1/conv2d/BiasAdd_grad/BiasAddGradBiasAddGrad5gradients_1/discriminator_1/conv2d/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
?
@gradients_1/discriminator_1/conv2d/BiasAdd_grad/tuple/group_depsNoOp<^gradients_1/discriminator_1/conv2d/BiasAdd_grad/BiasAddGrad6^gradients_1/discriminator_1/conv2d/Relu_grad/ReluGrad
?
Hgradients_1/discriminator_1/conv2d/BiasAdd_grad/tuple/control_dependencyIdentity5gradients_1/discriminator_1/conv2d/Relu_grad/ReluGradA^gradients_1/discriminator_1/conv2d/BiasAdd_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients_1/discriminator_1/conv2d/Relu_grad/ReluGrad*/
_output_shapes
:?????????@
?
Jgradients_1/discriminator_1/conv2d/BiasAdd_grad/tuple/control_dependency_1Identity;gradients_1/discriminator_1/conv2d/BiasAdd_grad/BiasAddGradA^gradients_1/discriminator_1/conv2d/BiasAdd_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients_1/discriminator_1/conv2d/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
?
5gradients_1/discriminator_1/conv2d/Conv2D_grad/ShapeNShapeN discriminator_1/dropout/Identity discriminator/conv2d/kernel/read*
T0*
out_type0*
N* 
_output_shapes
::
?
Bgradients_1/discriminator_1/conv2d/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput5gradients_1/discriminator_1/conv2d/Conv2D_grad/ShapeN discriminator/conv2d/kernel/readHgradients_1/discriminator_1/conv2d/BiasAdd_grad/tuple/control_dependency*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(*/
_output_shapes
:?????????  
?
Cgradients_1/discriminator_1/conv2d/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter discriminator_1/dropout/Identity7gradients_1/discriminator_1/conv2d/Conv2D_grad/ShapeN:1Hgradients_1/discriminator_1/conv2d/BiasAdd_grad/tuple/control_dependency*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(*&
_output_shapes
:@
?
?gradients_1/discriminator_1/conv2d/Conv2D_grad/tuple/group_depsNoOpD^gradients_1/discriminator_1/conv2d/Conv2D_grad/Conv2DBackpropFilterC^gradients_1/discriminator_1/conv2d/Conv2D_grad/Conv2DBackpropInput
?
Ggradients_1/discriminator_1/conv2d/Conv2D_grad/tuple/control_dependencyIdentityBgradients_1/discriminator_1/conv2d/Conv2D_grad/Conv2DBackpropInput@^gradients_1/discriminator_1/conv2d/Conv2D_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients_1/discriminator_1/conv2d/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:?????????  
?
Igradients_1/discriminator_1/conv2d/Conv2D_grad/tuple/control_dependency_1IdentityCgradients_1/discriminator_1/conv2d/Conv2D_grad/Conv2DBackpropFilter@^gradients_1/discriminator_1/conv2d/Conv2D_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients_1/discriminator_1/conv2d/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:@
?
(gradients_1/generator/Tanh_grad/TanhGradTanhGradgenerator/TanhGgradients_1/discriminator_1/conv2d/Conv2D_grad/tuple/control_dependency*
T0*/
_output_shapes
:?????????  
?
Agradients_1/generator/conv2d_transpose_2/BiasAdd_grad/BiasAddGradBiasAddGrad(gradients_1/generator/Tanh_grad/TanhGrad*
T0*
data_formatNHWC*
_output_shapes
:
?
Fgradients_1/generator/conv2d_transpose_2/BiasAdd_grad/tuple/group_depsNoOp)^gradients_1/generator/Tanh_grad/TanhGradB^gradients_1/generator/conv2d_transpose_2/BiasAdd_grad/BiasAddGrad
?
Ngradients_1/generator/conv2d_transpose_2/BiasAdd_grad/tuple/control_dependencyIdentity(gradients_1/generator/Tanh_grad/TanhGradG^gradients_1/generator/conv2d_transpose_2/BiasAdd_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_1/generator/Tanh_grad/TanhGrad*/
_output_shapes
:?????????  
?
Pgradients_1/generator/conv2d_transpose_2/BiasAdd_grad/tuple/control_dependency_1IdentityAgradients_1/generator/conv2d_transpose_2/BiasAdd_grad/BiasAddGradG^gradients_1/generator/conv2d_transpose_2/BiasAdd_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients_1/generator/conv2d_transpose_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
?
Dgradients_1/generator/conv2d_transpose_2/conv2d_transpose_grad/ShapeConst*%
valueB"             *
dtype0*
_output_shapes
:
?
Sgradients_1/generator/conv2d_transpose_2/conv2d_transpose_grad/Conv2DBackpropFilterConv2DBackpropFilterNgradients_1/generator/conv2d_transpose_2/BiasAdd_grad/tuple/control_dependencyDgradients_1/generator/conv2d_transpose_2/conv2d_transpose_grad/Shapegenerator/Maximum_2*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(*&
_output_shapes
: 
?
Egradients_1/generator/conv2d_transpose_2/conv2d_transpose_grad/Conv2DConv2DNgradients_1/generator/conv2d_transpose_2/BiasAdd_grad/tuple/control_dependency(generator/conv2d_transpose_2/kernel/read*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(*/
_output_shapes
:????????? 
?
Ogradients_1/generator/conv2d_transpose_2/conv2d_transpose_grad/tuple/group_depsNoOpF^gradients_1/generator/conv2d_transpose_2/conv2d_transpose_grad/Conv2DT^gradients_1/generator/conv2d_transpose_2/conv2d_transpose_grad/Conv2DBackpropFilter
?
Wgradients_1/generator/conv2d_transpose_2/conv2d_transpose_grad/tuple/control_dependencyIdentitySgradients_1/generator/conv2d_transpose_2/conv2d_transpose_grad/Conv2DBackpropFilterP^gradients_1/generator/conv2d_transpose_2/conv2d_transpose_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/generator/conv2d_transpose_2/conv2d_transpose_grad/Conv2DBackpropFilter*&
_output_shapes
: 
?
Ygradients_1/generator/conv2d_transpose_2/conv2d_transpose_grad/tuple/control_dependency_1IdentityEgradients_1/generator/conv2d_transpose_2/conv2d_transpose_grad/Conv2DP^gradients_1/generator/conv2d_transpose_2/conv2d_transpose_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients_1/generator/conv2d_transpose_2/conv2d_transpose_grad/Conv2D*/
_output_shapes
:????????? 
y
*gradients_1/generator/Maximum_2_grad/ShapeShapegenerator/mul_2*
T0*
out_type0*
_output_shapes
:
?
,gradients_1/generator/Maximum_2_grad/Shape_1Shape.generator/batch_normalization_2/FusedBatchNorm*
T0*
out_type0*
_output_shapes
:
?
,gradients_1/generator/Maximum_2_grad/Shape_2ShapeYgradients_1/generator/conv2d_transpose_2/conv2d_transpose_grad/tuple/control_dependency_1*
T0*
out_type0*
_output_shapes
:
u
0gradients_1/generator/Maximum_2_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
?
*gradients_1/generator/Maximum_2_grad/zerosFill,gradients_1/generator/Maximum_2_grad/Shape_20gradients_1/generator/Maximum_2_grad/zeros/Const*
T0*

index_type0*/
_output_shapes
:????????? 
?
1gradients_1/generator/Maximum_2_grad/GreaterEqualGreaterEqualgenerator/mul_2.generator/batch_normalization_2/FusedBatchNorm*
T0*/
_output_shapes
:????????? 
?
:gradients_1/generator/Maximum_2_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients_1/generator/Maximum_2_grad/Shape,gradients_1/generator/Maximum_2_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
+gradients_1/generator/Maximum_2_grad/SelectSelect1gradients_1/generator/Maximum_2_grad/GreaterEqualYgradients_1/generator/conv2d_transpose_2/conv2d_transpose_grad/tuple/control_dependency_1*gradients_1/generator/Maximum_2_grad/zeros*
T0*/
_output_shapes
:????????? 
?
-gradients_1/generator/Maximum_2_grad/Select_1Select1gradients_1/generator/Maximum_2_grad/GreaterEqual*gradients_1/generator/Maximum_2_grad/zerosYgradients_1/generator/conv2d_transpose_2/conv2d_transpose_grad/tuple/control_dependency_1*
T0*/
_output_shapes
:????????? 
?
(gradients_1/generator/Maximum_2_grad/SumSum+gradients_1/generator/Maximum_2_grad/Select:gradients_1/generator/Maximum_2_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
?
,gradients_1/generator/Maximum_2_grad/ReshapeReshape(gradients_1/generator/Maximum_2_grad/Sum*gradients_1/generator/Maximum_2_grad/Shape*
T0*
Tshape0*/
_output_shapes
:????????? 
?
*gradients_1/generator/Maximum_2_grad/Sum_1Sum-gradients_1/generator/Maximum_2_grad/Select_1<gradients_1/generator/Maximum_2_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
?
.gradients_1/generator/Maximum_2_grad/Reshape_1Reshape*gradients_1/generator/Maximum_2_grad/Sum_1,gradients_1/generator/Maximum_2_grad/Shape_1*
T0*
Tshape0*/
_output_shapes
:????????? 
?
5gradients_1/generator/Maximum_2_grad/tuple/group_depsNoOp-^gradients_1/generator/Maximum_2_grad/Reshape/^gradients_1/generator/Maximum_2_grad/Reshape_1
?
=gradients_1/generator/Maximum_2_grad/tuple/control_dependencyIdentity,gradients_1/generator/Maximum_2_grad/Reshape6^gradients_1/generator/Maximum_2_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/generator/Maximum_2_grad/Reshape*/
_output_shapes
:????????? 
?
?gradients_1/generator/Maximum_2_grad/tuple/control_dependency_1Identity.gradients_1/generator/Maximum_2_grad/Reshape_16^gradients_1/generator/Maximum_2_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/generator/Maximum_2_grad/Reshape_1*/
_output_shapes
:????????? 
i
&gradients_1/generator/mul_2_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
?
(gradients_1/generator/mul_2_grad/Shape_1Shape.generator/batch_normalization_2/FusedBatchNorm*
T0*
out_type0*
_output_shapes
:
?
6gradients_1/generator/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients_1/generator/mul_2_grad/Shape(gradients_1/generator/mul_2_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
$gradients_1/generator/mul_2_grad/MulMul=gradients_1/generator/Maximum_2_grad/tuple/control_dependency.generator/batch_normalization_2/FusedBatchNorm*
T0*/
_output_shapes
:????????? 
?
$gradients_1/generator/mul_2_grad/SumSum$gradients_1/generator/mul_2_grad/Mul6gradients_1/generator/mul_2_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
?
(gradients_1/generator/mul_2_grad/ReshapeReshape$gradients_1/generator/mul_2_grad/Sum&gradients_1/generator/mul_2_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
?
&gradients_1/generator/mul_2_grad/Mul_1Mulgenerator/mul_2/x=gradients_1/generator/Maximum_2_grad/tuple/control_dependency*
T0*/
_output_shapes
:????????? 
?
&gradients_1/generator/mul_2_grad/Sum_1Sum&gradients_1/generator/mul_2_grad/Mul_18gradients_1/generator/mul_2_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
?
*gradients_1/generator/mul_2_grad/Reshape_1Reshape&gradients_1/generator/mul_2_grad/Sum_1(gradients_1/generator/mul_2_grad/Shape_1*
T0*
Tshape0*/
_output_shapes
:????????? 
?
1gradients_1/generator/mul_2_grad/tuple/group_depsNoOp)^gradients_1/generator/mul_2_grad/Reshape+^gradients_1/generator/mul_2_grad/Reshape_1
?
9gradients_1/generator/mul_2_grad/tuple/control_dependencyIdentity(gradients_1/generator/mul_2_grad/Reshape2^gradients_1/generator/mul_2_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_1/generator/mul_2_grad/Reshape*
_output_shapes
: 
?
;gradients_1/generator/mul_2_grad/tuple/control_dependency_1Identity*gradients_1/generator/mul_2_grad/Reshape_12^gradients_1/generator/mul_2_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients_1/generator/mul_2_grad/Reshape_1*/
_output_shapes
:????????? 
?
gradients_1/AddNAddN?gradients_1/generator/Maximum_2_grad/tuple/control_dependency_1;gradients_1/generator/mul_2_grad/tuple/control_dependency_1*
T0*
N*A
_class7
53loc:@gradients_1/generator/Maximum_2_grad/Reshape_1*/
_output_shapes
:????????? 
}
gradients_1/zeros_like_12	ZerosLike0generator/batch_normalization_2/FusedBatchNorm:1*
T0*
_output_shapes
: 
}
gradients_1/zeros_like_13	ZerosLike0generator/batch_normalization_2/FusedBatchNorm:2*
T0*
_output_shapes
: 
}
gradients_1/zeros_like_14	ZerosLike0generator/batch_normalization_2/FusedBatchNorm:3*
T0*
_output_shapes
: 
}
gradients_1/zeros_like_15	ZerosLike0generator/batch_normalization_2/FusedBatchNorm:4*
T0*
_output_shapes
: 
?
Rgradients_1/generator/batch_normalization_2/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradgradients_1/AddN$generator/conv2d_transpose_1/BiasAdd*generator/batch_normalization_2/gamma/read0generator/batch_normalization_2/FusedBatchNorm:30generator/batch_normalization_2/FusedBatchNorm:4*
is_training(*
T0*
data_formatNHWC*
epsilon%o?:*C
_output_shapes1
/:????????? : : : : 
?
Pgradients_1/generator/batch_normalization_2/FusedBatchNorm_grad/tuple/group_depsNoOpS^gradients_1/generator/batch_normalization_2/FusedBatchNorm_grad/FusedBatchNormGrad
?
Xgradients_1/generator/batch_normalization_2/FusedBatchNorm_grad/tuple/control_dependencyIdentityRgradients_1/generator/batch_normalization_2/FusedBatchNorm_grad/FusedBatchNormGradQ^gradients_1/generator/batch_normalization_2/FusedBatchNorm_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/generator/batch_normalization_2/FusedBatchNorm_grad/FusedBatchNormGrad*/
_output_shapes
:????????? 
?
Zgradients_1/generator/batch_normalization_2/FusedBatchNorm_grad/tuple/control_dependency_1IdentityTgradients_1/generator/batch_normalization_2/FusedBatchNorm_grad/FusedBatchNormGrad:1Q^gradients_1/generator/batch_normalization_2/FusedBatchNorm_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/generator/batch_normalization_2/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
?
Zgradients_1/generator/batch_normalization_2/FusedBatchNorm_grad/tuple/control_dependency_2IdentityTgradients_1/generator/batch_normalization_2/FusedBatchNorm_grad/FusedBatchNormGrad:2Q^gradients_1/generator/batch_normalization_2/FusedBatchNorm_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/generator/batch_normalization_2/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
?
Zgradients_1/generator/batch_normalization_2/FusedBatchNorm_grad/tuple/control_dependency_3IdentityTgradients_1/generator/batch_normalization_2/FusedBatchNorm_grad/FusedBatchNormGrad:3Q^gradients_1/generator/batch_normalization_2/FusedBatchNorm_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/generator/batch_normalization_2/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
?
Zgradients_1/generator/batch_normalization_2/FusedBatchNorm_grad/tuple/control_dependency_4IdentityTgradients_1/generator/batch_normalization_2/FusedBatchNorm_grad/FusedBatchNormGrad:4Q^gradients_1/generator/batch_normalization_2/FusedBatchNorm_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/generator/batch_normalization_2/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
?
Agradients_1/generator/conv2d_transpose_1/BiasAdd_grad/BiasAddGradBiasAddGradXgradients_1/generator/batch_normalization_2/FusedBatchNorm_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes
: 
?
Fgradients_1/generator/conv2d_transpose_1/BiasAdd_grad/tuple/group_depsNoOpY^gradients_1/generator/batch_normalization_2/FusedBatchNorm_grad/tuple/control_dependencyB^gradients_1/generator/conv2d_transpose_1/BiasAdd_grad/BiasAddGrad
?
Ngradients_1/generator/conv2d_transpose_1/BiasAdd_grad/tuple/control_dependencyIdentityXgradients_1/generator/batch_normalization_2/FusedBatchNorm_grad/tuple/control_dependencyG^gradients_1/generator/conv2d_transpose_1/BiasAdd_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/generator/batch_normalization_2/FusedBatchNorm_grad/FusedBatchNormGrad*/
_output_shapes
:????????? 
?
Pgradients_1/generator/conv2d_transpose_1/BiasAdd_grad/tuple/control_dependency_1IdentityAgradients_1/generator/conv2d_transpose_1/BiasAdd_grad/BiasAddGradG^gradients_1/generator/conv2d_transpose_1/BiasAdd_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients_1/generator/conv2d_transpose_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
?
Dgradients_1/generator/conv2d_transpose_1/conv2d_transpose_grad/ShapeConst*%
valueB"          @   *
dtype0*
_output_shapes
:
?
Sgradients_1/generator/conv2d_transpose_1/conv2d_transpose_grad/Conv2DBackpropFilterConv2DBackpropFilterNgradients_1/generator/conv2d_transpose_1/BiasAdd_grad/tuple/control_dependencyDgradients_1/generator/conv2d_transpose_1/conv2d_transpose_grad/Shapegenerator/Maximum_1*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(*&
_output_shapes
: @
?
Egradients_1/generator/conv2d_transpose_1/conv2d_transpose_grad/Conv2DConv2DNgradients_1/generator/conv2d_transpose_1/BiasAdd_grad/tuple/control_dependency(generator/conv2d_transpose_1/kernel/read*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(*/
_output_shapes
:?????????@
?
Ogradients_1/generator/conv2d_transpose_1/conv2d_transpose_grad/tuple/group_depsNoOpF^gradients_1/generator/conv2d_transpose_1/conv2d_transpose_grad/Conv2DT^gradients_1/generator/conv2d_transpose_1/conv2d_transpose_grad/Conv2DBackpropFilter
?
Wgradients_1/generator/conv2d_transpose_1/conv2d_transpose_grad/tuple/control_dependencyIdentitySgradients_1/generator/conv2d_transpose_1/conv2d_transpose_grad/Conv2DBackpropFilterP^gradients_1/generator/conv2d_transpose_1/conv2d_transpose_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/generator/conv2d_transpose_1/conv2d_transpose_grad/Conv2DBackpropFilter*&
_output_shapes
: @
?
Ygradients_1/generator/conv2d_transpose_1/conv2d_transpose_grad/tuple/control_dependency_1IdentityEgradients_1/generator/conv2d_transpose_1/conv2d_transpose_grad/Conv2DP^gradients_1/generator/conv2d_transpose_1/conv2d_transpose_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients_1/generator/conv2d_transpose_1/conv2d_transpose_grad/Conv2D*/
_output_shapes
:?????????@
y
*gradients_1/generator/Maximum_1_grad/ShapeShapegenerator/mul_1*
T0*
out_type0*
_output_shapes
:
?
,gradients_1/generator/Maximum_1_grad/Shape_1Shape.generator/batch_normalization_1/FusedBatchNorm*
T0*
out_type0*
_output_shapes
:
?
,gradients_1/generator/Maximum_1_grad/Shape_2ShapeYgradients_1/generator/conv2d_transpose_1/conv2d_transpose_grad/tuple/control_dependency_1*
T0*
out_type0*
_output_shapes
:
u
0gradients_1/generator/Maximum_1_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
?
*gradients_1/generator/Maximum_1_grad/zerosFill,gradients_1/generator/Maximum_1_grad/Shape_20gradients_1/generator/Maximum_1_grad/zeros/Const*
T0*

index_type0*/
_output_shapes
:?????????@
?
1gradients_1/generator/Maximum_1_grad/GreaterEqualGreaterEqualgenerator/mul_1.generator/batch_normalization_1/FusedBatchNorm*
T0*/
_output_shapes
:?????????@
?
:gradients_1/generator/Maximum_1_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients_1/generator/Maximum_1_grad/Shape,gradients_1/generator/Maximum_1_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
+gradients_1/generator/Maximum_1_grad/SelectSelect1gradients_1/generator/Maximum_1_grad/GreaterEqualYgradients_1/generator/conv2d_transpose_1/conv2d_transpose_grad/tuple/control_dependency_1*gradients_1/generator/Maximum_1_grad/zeros*
T0*/
_output_shapes
:?????????@
?
-gradients_1/generator/Maximum_1_grad/Select_1Select1gradients_1/generator/Maximum_1_grad/GreaterEqual*gradients_1/generator/Maximum_1_grad/zerosYgradients_1/generator/conv2d_transpose_1/conv2d_transpose_grad/tuple/control_dependency_1*
T0*/
_output_shapes
:?????????@
?
(gradients_1/generator/Maximum_1_grad/SumSum+gradients_1/generator/Maximum_1_grad/Select:gradients_1/generator/Maximum_1_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
?
,gradients_1/generator/Maximum_1_grad/ReshapeReshape(gradients_1/generator/Maximum_1_grad/Sum*gradients_1/generator/Maximum_1_grad/Shape*
T0*
Tshape0*/
_output_shapes
:?????????@
?
*gradients_1/generator/Maximum_1_grad/Sum_1Sum-gradients_1/generator/Maximum_1_grad/Select_1<gradients_1/generator/Maximum_1_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
?
.gradients_1/generator/Maximum_1_grad/Reshape_1Reshape*gradients_1/generator/Maximum_1_grad/Sum_1,gradients_1/generator/Maximum_1_grad/Shape_1*
T0*
Tshape0*/
_output_shapes
:?????????@
?
5gradients_1/generator/Maximum_1_grad/tuple/group_depsNoOp-^gradients_1/generator/Maximum_1_grad/Reshape/^gradients_1/generator/Maximum_1_grad/Reshape_1
?
=gradients_1/generator/Maximum_1_grad/tuple/control_dependencyIdentity,gradients_1/generator/Maximum_1_grad/Reshape6^gradients_1/generator/Maximum_1_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/generator/Maximum_1_grad/Reshape*/
_output_shapes
:?????????@
?
?gradients_1/generator/Maximum_1_grad/tuple/control_dependency_1Identity.gradients_1/generator/Maximum_1_grad/Reshape_16^gradients_1/generator/Maximum_1_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/generator/Maximum_1_grad/Reshape_1*/
_output_shapes
:?????????@
i
&gradients_1/generator/mul_1_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
?
(gradients_1/generator/mul_1_grad/Shape_1Shape.generator/batch_normalization_1/FusedBatchNorm*
T0*
out_type0*
_output_shapes
:
?
6gradients_1/generator/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients_1/generator/mul_1_grad/Shape(gradients_1/generator/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
$gradients_1/generator/mul_1_grad/MulMul=gradients_1/generator/Maximum_1_grad/tuple/control_dependency.generator/batch_normalization_1/FusedBatchNorm*
T0*/
_output_shapes
:?????????@
?
$gradients_1/generator/mul_1_grad/SumSum$gradients_1/generator/mul_1_grad/Mul6gradients_1/generator/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
?
(gradients_1/generator/mul_1_grad/ReshapeReshape$gradients_1/generator/mul_1_grad/Sum&gradients_1/generator/mul_1_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
?
&gradients_1/generator/mul_1_grad/Mul_1Mulgenerator/mul_1/x=gradients_1/generator/Maximum_1_grad/tuple/control_dependency*
T0*/
_output_shapes
:?????????@
?
&gradients_1/generator/mul_1_grad/Sum_1Sum&gradients_1/generator/mul_1_grad/Mul_18gradients_1/generator/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
?
*gradients_1/generator/mul_1_grad/Reshape_1Reshape&gradients_1/generator/mul_1_grad/Sum_1(gradients_1/generator/mul_1_grad/Shape_1*
T0*
Tshape0*/
_output_shapes
:?????????@
?
1gradients_1/generator/mul_1_grad/tuple/group_depsNoOp)^gradients_1/generator/mul_1_grad/Reshape+^gradients_1/generator/mul_1_grad/Reshape_1
?
9gradients_1/generator/mul_1_grad/tuple/control_dependencyIdentity(gradients_1/generator/mul_1_grad/Reshape2^gradients_1/generator/mul_1_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_1/generator/mul_1_grad/Reshape*
_output_shapes
: 
?
;gradients_1/generator/mul_1_grad/tuple/control_dependency_1Identity*gradients_1/generator/mul_1_grad/Reshape_12^gradients_1/generator/mul_1_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients_1/generator/mul_1_grad/Reshape_1*/
_output_shapes
:?????????@
?
gradients_1/AddN_1AddN?gradients_1/generator/Maximum_1_grad/tuple/control_dependency_1;gradients_1/generator/mul_1_grad/tuple/control_dependency_1*
T0*
N*A
_class7
53loc:@gradients_1/generator/Maximum_1_grad/Reshape_1*/
_output_shapes
:?????????@
}
gradients_1/zeros_like_16	ZerosLike0generator/batch_normalization_1/FusedBatchNorm:1*
T0*
_output_shapes
:@
}
gradients_1/zeros_like_17	ZerosLike0generator/batch_normalization_1/FusedBatchNorm:2*
T0*
_output_shapes
:@
}
gradients_1/zeros_like_18	ZerosLike0generator/batch_normalization_1/FusedBatchNorm:3*
T0*
_output_shapes
:@
}
gradients_1/zeros_like_19	ZerosLike0generator/batch_normalization_1/FusedBatchNorm:4*
T0*
_output_shapes
:@
?
Rgradients_1/generator/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradgradients_1/AddN_1"generator/conv2d_transpose/BiasAdd*generator/batch_normalization_1/gamma/read0generator/batch_normalization_1/FusedBatchNorm:30generator/batch_normalization_1/FusedBatchNorm:4*
is_training(*
T0*
data_formatNHWC*
epsilon%o?:*C
_output_shapes1
/:?????????@:@:@: : 
?
Pgradients_1/generator/batch_normalization_1/FusedBatchNorm_grad/tuple/group_depsNoOpS^gradients_1/generator/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGrad
?
Xgradients_1/generator/batch_normalization_1/FusedBatchNorm_grad/tuple/control_dependencyIdentityRgradients_1/generator/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGradQ^gradients_1/generator/batch_normalization_1/FusedBatchNorm_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/generator/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGrad*/
_output_shapes
:?????????@
?
Zgradients_1/generator/batch_normalization_1/FusedBatchNorm_grad/tuple/control_dependency_1IdentityTgradients_1/generator/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGrad:1Q^gradients_1/generator/batch_normalization_1/FusedBatchNorm_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/generator/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
:@
?
Zgradients_1/generator/batch_normalization_1/FusedBatchNorm_grad/tuple/control_dependency_2IdentityTgradients_1/generator/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGrad:2Q^gradients_1/generator/batch_normalization_1/FusedBatchNorm_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/generator/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
:@
?
Zgradients_1/generator/batch_normalization_1/FusedBatchNorm_grad/tuple/control_dependency_3IdentityTgradients_1/generator/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGrad:3Q^gradients_1/generator/batch_normalization_1/FusedBatchNorm_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/generator/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
?
Zgradients_1/generator/batch_normalization_1/FusedBatchNorm_grad/tuple/control_dependency_4IdentityTgradients_1/generator/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGrad:4Q^gradients_1/generator/batch_normalization_1/FusedBatchNorm_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/generator/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
?
?gradients_1/generator/conv2d_transpose/BiasAdd_grad/BiasAddGradBiasAddGradXgradients_1/generator/batch_normalization_1/FusedBatchNorm_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes
:@
?
Dgradients_1/generator/conv2d_transpose/BiasAdd_grad/tuple/group_depsNoOpY^gradients_1/generator/batch_normalization_1/FusedBatchNorm_grad/tuple/control_dependency@^gradients_1/generator/conv2d_transpose/BiasAdd_grad/BiasAddGrad
?
Lgradients_1/generator/conv2d_transpose/BiasAdd_grad/tuple/control_dependencyIdentityXgradients_1/generator/batch_normalization_1/FusedBatchNorm_grad/tuple/control_dependencyE^gradients_1/generator/conv2d_transpose/BiasAdd_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/generator/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGrad*/
_output_shapes
:?????????@
?
Ngradients_1/generator/conv2d_transpose/BiasAdd_grad/tuple/control_dependency_1Identity?gradients_1/generator/conv2d_transpose/BiasAdd_grad/BiasAddGradE^gradients_1/generator/conv2d_transpose/BiasAdd_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients_1/generator/conv2d_transpose/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
?
Bgradients_1/generator/conv2d_transpose/conv2d_transpose_grad/ShapeConst*%
valueB"      @   ?   *
dtype0*
_output_shapes
:
?
Qgradients_1/generator/conv2d_transpose/conv2d_transpose_grad/Conv2DBackpropFilterConv2DBackpropFilterLgradients_1/generator/conv2d_transpose/BiasAdd_grad/tuple/control_dependencyBgradients_1/generator/conv2d_transpose/conv2d_transpose_grad/Shapegenerator/Maximum*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(*'
_output_shapes
:@?
?
Cgradients_1/generator/conv2d_transpose/conv2d_transpose_grad/Conv2DConv2DLgradients_1/generator/conv2d_transpose/BiasAdd_grad/tuple/control_dependency&generator/conv2d_transpose/kernel/read*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(*0
_output_shapes
:??????????
?
Mgradients_1/generator/conv2d_transpose/conv2d_transpose_grad/tuple/group_depsNoOpD^gradients_1/generator/conv2d_transpose/conv2d_transpose_grad/Conv2DR^gradients_1/generator/conv2d_transpose/conv2d_transpose_grad/Conv2DBackpropFilter
?
Ugradients_1/generator/conv2d_transpose/conv2d_transpose_grad/tuple/control_dependencyIdentityQgradients_1/generator/conv2d_transpose/conv2d_transpose_grad/Conv2DBackpropFilterN^gradients_1/generator/conv2d_transpose/conv2d_transpose_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients_1/generator/conv2d_transpose/conv2d_transpose_grad/Conv2DBackpropFilter*'
_output_shapes
:@?
?
Wgradients_1/generator/conv2d_transpose/conv2d_transpose_grad/tuple/control_dependency_1IdentityCgradients_1/generator/conv2d_transpose/conv2d_transpose_grad/Conv2DN^gradients_1/generator/conv2d_transpose/conv2d_transpose_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients_1/generator/conv2d_transpose/conv2d_transpose_grad/Conv2D*0
_output_shapes
:??????????
u
(gradients_1/generator/Maximum_grad/ShapeShapegenerator/mul*
T0*
out_type0*
_output_shapes
:
?
*gradients_1/generator/Maximum_grad/Shape_1Shape,generator/batch_normalization/FusedBatchNorm*
T0*
out_type0*
_output_shapes
:
?
*gradients_1/generator/Maximum_grad/Shape_2ShapeWgradients_1/generator/conv2d_transpose/conv2d_transpose_grad/tuple/control_dependency_1*
T0*
out_type0*
_output_shapes
:
s
.gradients_1/generator/Maximum_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
?
(gradients_1/generator/Maximum_grad/zerosFill*gradients_1/generator/Maximum_grad/Shape_2.gradients_1/generator/Maximum_grad/zeros/Const*
T0*

index_type0*0
_output_shapes
:??????????
?
/gradients_1/generator/Maximum_grad/GreaterEqualGreaterEqualgenerator/mul,generator/batch_normalization/FusedBatchNorm*
T0*0
_output_shapes
:??????????
?
8gradients_1/generator/Maximum_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients_1/generator/Maximum_grad/Shape*gradients_1/generator/Maximum_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
)gradients_1/generator/Maximum_grad/SelectSelect/gradients_1/generator/Maximum_grad/GreaterEqualWgradients_1/generator/conv2d_transpose/conv2d_transpose_grad/tuple/control_dependency_1(gradients_1/generator/Maximum_grad/zeros*
T0*0
_output_shapes
:??????????
?
+gradients_1/generator/Maximum_grad/Select_1Select/gradients_1/generator/Maximum_grad/GreaterEqual(gradients_1/generator/Maximum_grad/zerosWgradients_1/generator/conv2d_transpose/conv2d_transpose_grad/tuple/control_dependency_1*
T0*0
_output_shapes
:??????????
?
&gradients_1/generator/Maximum_grad/SumSum)gradients_1/generator/Maximum_grad/Select8gradients_1/generator/Maximum_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
?
*gradients_1/generator/Maximum_grad/ReshapeReshape&gradients_1/generator/Maximum_grad/Sum(gradients_1/generator/Maximum_grad/Shape*
T0*
Tshape0*0
_output_shapes
:??????????
?
(gradients_1/generator/Maximum_grad/Sum_1Sum+gradients_1/generator/Maximum_grad/Select_1:gradients_1/generator/Maximum_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
?
,gradients_1/generator/Maximum_grad/Reshape_1Reshape(gradients_1/generator/Maximum_grad/Sum_1*gradients_1/generator/Maximum_grad/Shape_1*
T0*
Tshape0*0
_output_shapes
:??????????
?
3gradients_1/generator/Maximum_grad/tuple/group_depsNoOp+^gradients_1/generator/Maximum_grad/Reshape-^gradients_1/generator/Maximum_grad/Reshape_1
?
;gradients_1/generator/Maximum_grad/tuple/control_dependencyIdentity*gradients_1/generator/Maximum_grad/Reshape4^gradients_1/generator/Maximum_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients_1/generator/Maximum_grad/Reshape*0
_output_shapes
:??????????
?
=gradients_1/generator/Maximum_grad/tuple/control_dependency_1Identity,gradients_1/generator/Maximum_grad/Reshape_14^gradients_1/generator/Maximum_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/generator/Maximum_grad/Reshape_1*0
_output_shapes
:??????????
g
$gradients_1/generator/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
?
&gradients_1/generator/mul_grad/Shape_1Shape,generator/batch_normalization/FusedBatchNorm*
T0*
out_type0*
_output_shapes
:
?
4gradients_1/generator/mul_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients_1/generator/mul_grad/Shape&gradients_1/generator/mul_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
"gradients_1/generator/mul_grad/MulMul;gradients_1/generator/Maximum_grad/tuple/control_dependency,generator/batch_normalization/FusedBatchNorm*
T0*0
_output_shapes
:??????????
?
"gradients_1/generator/mul_grad/SumSum"gradients_1/generator/mul_grad/Mul4gradients_1/generator/mul_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
?
&gradients_1/generator/mul_grad/ReshapeReshape"gradients_1/generator/mul_grad/Sum$gradients_1/generator/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
?
$gradients_1/generator/mul_grad/Mul_1Mulgenerator/mul/x;gradients_1/generator/Maximum_grad/tuple/control_dependency*
T0*0
_output_shapes
:??????????
?
$gradients_1/generator/mul_grad/Sum_1Sum$gradients_1/generator/mul_grad/Mul_16gradients_1/generator/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
?
(gradients_1/generator/mul_grad/Reshape_1Reshape$gradients_1/generator/mul_grad/Sum_1&gradients_1/generator/mul_grad/Shape_1*
T0*
Tshape0*0
_output_shapes
:??????????
?
/gradients_1/generator/mul_grad/tuple/group_depsNoOp'^gradients_1/generator/mul_grad/Reshape)^gradients_1/generator/mul_grad/Reshape_1
?
7gradients_1/generator/mul_grad/tuple/control_dependencyIdentity&gradients_1/generator/mul_grad/Reshape0^gradients_1/generator/mul_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients_1/generator/mul_grad/Reshape*
_output_shapes
: 
?
9gradients_1/generator/mul_grad/tuple/control_dependency_1Identity(gradients_1/generator/mul_grad/Reshape_10^gradients_1/generator/mul_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_1/generator/mul_grad/Reshape_1*0
_output_shapes
:??????????
?
gradients_1/AddN_2AddN=gradients_1/generator/Maximum_grad/tuple/control_dependency_19gradients_1/generator/mul_grad/tuple/control_dependency_1*
T0*
N*?
_class5
31loc:@gradients_1/generator/Maximum_grad/Reshape_1*0
_output_shapes
:??????????
|
gradients_1/zeros_like_20	ZerosLike.generator/batch_normalization/FusedBatchNorm:1*
T0*
_output_shapes	
:?
|
gradients_1/zeros_like_21	ZerosLike.generator/batch_normalization/FusedBatchNorm:2*
T0*
_output_shapes	
:?
|
gradients_1/zeros_like_22	ZerosLike.generator/batch_normalization/FusedBatchNorm:3*
T0*
_output_shapes	
:?
|
gradients_1/zeros_like_23	ZerosLike.generator/batch_normalization/FusedBatchNorm:4*
T0*
_output_shapes	
:?
?
Pgradients_1/generator/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradgradients_1/AddN_2generator/Reshape(generator/batch_normalization/gamma/read.generator/batch_normalization/FusedBatchNorm:3.generator/batch_normalization/FusedBatchNorm:4*
is_training(*
T0*
data_formatNHWC*
epsilon%o?:*F
_output_shapes4
2:??????????:?:?: : 
?
Ngradients_1/generator/batch_normalization/FusedBatchNorm_grad/tuple/group_depsNoOpQ^gradients_1/generator/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGrad
?
Vgradients_1/generator/batch_normalization/FusedBatchNorm_grad/tuple/control_dependencyIdentityPgradients_1/generator/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGradO^gradients_1/generator/batch_normalization/FusedBatchNorm_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients_1/generator/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGrad*0
_output_shapes
:??????????
?
Xgradients_1/generator/batch_normalization/FusedBatchNorm_grad/tuple/control_dependency_1IdentityRgradients_1/generator/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGrad:1O^gradients_1/generator/batch_normalization/FusedBatchNorm_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients_1/generator/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes	
:?
?
Xgradients_1/generator/batch_normalization/FusedBatchNorm_grad/tuple/control_dependency_2IdentityRgradients_1/generator/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGrad:2O^gradients_1/generator/batch_normalization/FusedBatchNorm_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients_1/generator/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes	
:?
?
Xgradients_1/generator/batch_normalization/FusedBatchNorm_grad/tuple/control_dependency_3IdentityRgradients_1/generator/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGrad:3O^gradients_1/generator/batch_normalization/FusedBatchNorm_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients_1/generator/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
?
Xgradients_1/generator/batch_normalization/FusedBatchNorm_grad/tuple/control_dependency_4IdentityRgradients_1/generator/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGrad:4O^gradients_1/generator/batch_normalization/FusedBatchNorm_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients_1/generator/batch_normalization/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 

(gradients_1/generator/Reshape_grad/ShapeShapegenerator/dense/BiasAdd*
T0*
out_type0*
_output_shapes
:
?
*gradients_1/generator/Reshape_grad/ReshapeReshapeVgradients_1/generator/batch_normalization/FusedBatchNorm_grad/tuple/control_dependency(gradients_1/generator/Reshape_grad/Shape*
T0*
Tshape0*(
_output_shapes
:??????????
?
4gradients_1/generator/dense/BiasAdd_grad/BiasAddGradBiasAddGrad*gradients_1/generator/Reshape_grad/Reshape*
T0*
data_formatNHWC*
_output_shapes	
:?
?
9gradients_1/generator/dense/BiasAdd_grad/tuple/group_depsNoOp+^gradients_1/generator/Reshape_grad/Reshape5^gradients_1/generator/dense/BiasAdd_grad/BiasAddGrad
?
Agradients_1/generator/dense/BiasAdd_grad/tuple/control_dependencyIdentity*gradients_1/generator/Reshape_grad/Reshape:^gradients_1/generator/dense/BiasAdd_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients_1/generator/Reshape_grad/Reshape*(
_output_shapes
:??????????
?
Cgradients_1/generator/dense/BiasAdd_grad/tuple/control_dependency_1Identity4gradients_1/generator/dense/BiasAdd_grad/BiasAddGrad:^gradients_1/generator/dense/BiasAdd_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients_1/generator/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:?
?
.gradients_1/generator/dense/MatMul_grad/MatMulMatMulAgradients_1/generator/dense/BiasAdd_grad/tuple/control_dependencygenerator/dense/kernel/read*
T0*
transpose_b(*
transpose_a( *'
_output_shapes
:?????????d
?
0gradients_1/generator/dense/MatMul_grad/MatMul_1MatMulinput_zAgradients_1/generator/dense/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_b( *
transpose_a(*
_output_shapes
:	d?
?
8gradients_1/generator/dense/MatMul_grad/tuple/group_depsNoOp/^gradients_1/generator/dense/MatMul_grad/MatMul1^gradients_1/generator/dense/MatMul_grad/MatMul_1
?
@gradients_1/generator/dense/MatMul_grad/tuple/control_dependencyIdentity.gradients_1/generator/dense/MatMul_grad/MatMul9^gradients_1/generator/dense/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/generator/dense/MatMul_grad/MatMul*'
_output_shapes
:?????????d
?
Bgradients_1/generator/dense/MatMul_grad/tuple/control_dependency_1Identity0gradients_1/generator/dense/MatMul_grad/MatMul_19^gradients_1/generator/dense/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_1/generator/dense/MatMul_grad/MatMul_1*
_output_shapes
:	d?
?
beta1_power_1/initial_valueConst*
valueB
 *   ?*
dtype0*5
_class+
)'loc:@generator/batch_normalization/beta*
_output_shapes
: 
?
beta1_power_1
VariableV2*
dtype0*
shared_name *
shape: *
	container *5
_class+
)'loc:@generator/batch_normalization/beta*
_output_shapes
: 
?
beta1_power_1/AssignAssignbeta1_power_1beta1_power_1/initial_value*
T0*
use_locking(*
validate_shape(*5
_class+
)'loc:@generator/batch_normalization/beta*
_output_shapes
: 
?
beta1_power_1/readIdentitybeta1_power_1*
T0*5
_class+
)'loc:@generator/batch_normalization/beta*
_output_shapes
: 
?
beta2_power_1/initial_valueConst*
valueB
 *w??*
dtype0*5
_class+
)'loc:@generator/batch_normalization/beta*
_output_shapes
: 
?
beta2_power_1
VariableV2*
dtype0*
shared_name *
shape: *
	container *5
_class+
)'loc:@generator/batch_normalization/beta*
_output_shapes
: 
?
beta2_power_1/AssignAssignbeta2_power_1beta2_power_1/initial_value*
T0*
use_locking(*
validate_shape(*5
_class+
)'loc:@generator/batch_normalization/beta*
_output_shapes
: 
?
beta2_power_1/readIdentitybeta2_power_1*
T0*5
_class+
)'loc:@generator/batch_normalization/beta*
_output_shapes
: 
?
=generator/dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"d      *
dtype0*)
_class
loc:@generator/dense/kernel*
_output_shapes
:
?
3generator/dense/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*)
_class
loc:@generator/dense/kernel*
_output_shapes
: 
?
-generator/dense/kernel/Adam/Initializer/zerosFill=generator/dense/kernel/Adam/Initializer/zeros/shape_as_tensor3generator/dense/kernel/Adam/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@generator/dense/kernel*
_output_shapes
:	d?
?
generator/dense/kernel/Adam
VariableV2*
dtype0*
shared_name *
shape:	d?*
	container *)
_class
loc:@generator/dense/kernel*
_output_shapes
:	d?
?
"generator/dense/kernel/Adam/AssignAssigngenerator/dense/kernel/Adam-generator/dense/kernel/Adam/Initializer/zeros*
T0*
use_locking(*
validate_shape(*)
_class
loc:@generator/dense/kernel*
_output_shapes
:	d?
?
 generator/dense/kernel/Adam/readIdentitygenerator/dense/kernel/Adam*
T0*)
_class
loc:@generator/dense/kernel*
_output_shapes
:	d?
?
?generator/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"d      *
dtype0*)
_class
loc:@generator/dense/kernel*
_output_shapes
:
?
5generator/dense/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*)
_class
loc:@generator/dense/kernel*
_output_shapes
: 
?
/generator/dense/kernel/Adam_1/Initializer/zerosFill?generator/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor5generator/dense/kernel/Adam_1/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@generator/dense/kernel*
_output_shapes
:	d?
?
generator/dense/kernel/Adam_1
VariableV2*
dtype0*
shared_name *
shape:	d?*
	container *)
_class
loc:@generator/dense/kernel*
_output_shapes
:	d?
?
$generator/dense/kernel/Adam_1/AssignAssigngenerator/dense/kernel/Adam_1/generator/dense/kernel/Adam_1/Initializer/zeros*
T0*
use_locking(*
validate_shape(*)
_class
loc:@generator/dense/kernel*
_output_shapes
:	d?
?
"generator/dense/kernel/Adam_1/readIdentitygenerator/dense/kernel/Adam_1*
T0*)
_class
loc:@generator/dense/kernel*
_output_shapes
:	d?
?
;generator/dense/bias/Adam/Initializer/zeros/shape_as_tensorConst*
valueB:?*
dtype0*'
_class
loc:@generator/dense/bias*
_output_shapes
:
?
1generator/dense/bias/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*'
_class
loc:@generator/dense/bias*
_output_shapes
: 
?
+generator/dense/bias/Adam/Initializer/zerosFill;generator/dense/bias/Adam/Initializer/zeros/shape_as_tensor1generator/dense/bias/Adam/Initializer/zeros/Const*
T0*

index_type0*'
_class
loc:@generator/dense/bias*
_output_shapes	
:?
?
generator/dense/bias/Adam
VariableV2*
dtype0*
shared_name *
shape:?*
	container *'
_class
loc:@generator/dense/bias*
_output_shapes	
:?
?
 generator/dense/bias/Adam/AssignAssigngenerator/dense/bias/Adam+generator/dense/bias/Adam/Initializer/zeros*
T0*
use_locking(*
validate_shape(*'
_class
loc:@generator/dense/bias*
_output_shapes	
:?
?
generator/dense/bias/Adam/readIdentitygenerator/dense/bias/Adam*
T0*'
_class
loc:@generator/dense/bias*
_output_shapes	
:?
?
=generator/dense/bias/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB:?*
dtype0*'
_class
loc:@generator/dense/bias*
_output_shapes
:
?
3generator/dense/bias/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*'
_class
loc:@generator/dense/bias*
_output_shapes
: 
?
-generator/dense/bias/Adam_1/Initializer/zerosFill=generator/dense/bias/Adam_1/Initializer/zeros/shape_as_tensor3generator/dense/bias/Adam_1/Initializer/zeros/Const*
T0*

index_type0*'
_class
loc:@generator/dense/bias*
_output_shapes	
:?
?
generator/dense/bias/Adam_1
VariableV2*
dtype0*
shared_name *
shape:?*
	container *'
_class
loc:@generator/dense/bias*
_output_shapes	
:?
?
"generator/dense/bias/Adam_1/AssignAssigngenerator/dense/bias/Adam_1-generator/dense/bias/Adam_1/Initializer/zeros*
T0*
use_locking(*
validate_shape(*'
_class
loc:@generator/dense/bias*
_output_shapes	
:?
?
 generator/dense/bias/Adam_1/readIdentitygenerator/dense/bias/Adam_1*
T0*'
_class
loc:@generator/dense/bias*
_output_shapes	
:?
?
:generator/batch_normalization/gamma/Adam/Initializer/zerosConst*
valueB?*    *
dtype0*6
_class,
*(loc:@generator/batch_normalization/gamma*
_output_shapes	
:?
?
(generator/batch_normalization/gamma/Adam
VariableV2*
dtype0*
shared_name *
shape:?*
	container *6
_class,
*(loc:@generator/batch_normalization/gamma*
_output_shapes	
:?
?
/generator/batch_normalization/gamma/Adam/AssignAssign(generator/batch_normalization/gamma/Adam:generator/batch_normalization/gamma/Adam/Initializer/zeros*
T0*
use_locking(*
validate_shape(*6
_class,
*(loc:@generator/batch_normalization/gamma*
_output_shapes	
:?
?
-generator/batch_normalization/gamma/Adam/readIdentity(generator/batch_normalization/gamma/Adam*
T0*6
_class,
*(loc:@generator/batch_normalization/gamma*
_output_shapes	
:?
?
<generator/batch_normalization/gamma/Adam_1/Initializer/zerosConst*
valueB?*    *
dtype0*6
_class,
*(loc:@generator/batch_normalization/gamma*
_output_shapes	
:?
?
*generator/batch_normalization/gamma/Adam_1
VariableV2*
dtype0*
shared_name *
shape:?*
	container *6
_class,
*(loc:@generator/batch_normalization/gamma*
_output_shapes	
:?
?
1generator/batch_normalization/gamma/Adam_1/AssignAssign*generator/batch_normalization/gamma/Adam_1<generator/batch_normalization/gamma/Adam_1/Initializer/zeros*
T0*
use_locking(*
validate_shape(*6
_class,
*(loc:@generator/batch_normalization/gamma*
_output_shapes	
:?
?
/generator/batch_normalization/gamma/Adam_1/readIdentity*generator/batch_normalization/gamma/Adam_1*
T0*6
_class,
*(loc:@generator/batch_normalization/gamma*
_output_shapes	
:?
?
9generator/batch_normalization/beta/Adam/Initializer/zerosConst*
valueB?*    *
dtype0*5
_class+
)'loc:@generator/batch_normalization/beta*
_output_shapes	
:?
?
'generator/batch_normalization/beta/Adam
VariableV2*
dtype0*
shared_name *
shape:?*
	container *5
_class+
)'loc:@generator/batch_normalization/beta*
_output_shapes	
:?
?
.generator/batch_normalization/beta/Adam/AssignAssign'generator/batch_normalization/beta/Adam9generator/batch_normalization/beta/Adam/Initializer/zeros*
T0*
use_locking(*
validate_shape(*5
_class+
)'loc:@generator/batch_normalization/beta*
_output_shapes	
:?
?
,generator/batch_normalization/beta/Adam/readIdentity'generator/batch_normalization/beta/Adam*
T0*5
_class+
)'loc:@generator/batch_normalization/beta*
_output_shapes	
:?
?
;generator/batch_normalization/beta/Adam_1/Initializer/zerosConst*
valueB?*    *
dtype0*5
_class+
)'loc:@generator/batch_normalization/beta*
_output_shapes	
:?
?
)generator/batch_normalization/beta/Adam_1
VariableV2*
dtype0*
shared_name *
shape:?*
	container *5
_class+
)'loc:@generator/batch_normalization/beta*
_output_shapes	
:?
?
0generator/batch_normalization/beta/Adam_1/AssignAssign)generator/batch_normalization/beta/Adam_1;generator/batch_normalization/beta/Adam_1/Initializer/zeros*
T0*
use_locking(*
validate_shape(*5
_class+
)'loc:@generator/batch_normalization/beta*
_output_shapes	
:?
?
.generator/batch_normalization/beta/Adam_1/readIdentity)generator/batch_normalization/beta/Adam_1*
T0*5
_class+
)'loc:@generator/batch_normalization/beta*
_output_shapes	
:?
?
Hgenerator/conv2d_transpose/kernel/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"      @   ?   *
dtype0*4
_class*
(&loc:@generator/conv2d_transpose/kernel*
_output_shapes
:
?
>generator/conv2d_transpose/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*4
_class*
(&loc:@generator/conv2d_transpose/kernel*
_output_shapes
: 
?
8generator/conv2d_transpose/kernel/Adam/Initializer/zerosFillHgenerator/conv2d_transpose/kernel/Adam/Initializer/zeros/shape_as_tensor>generator/conv2d_transpose/kernel/Adam/Initializer/zeros/Const*
T0*

index_type0*4
_class*
(&loc:@generator/conv2d_transpose/kernel*'
_output_shapes
:@?
?
&generator/conv2d_transpose/kernel/Adam
VariableV2*
dtype0*
shared_name *
shape:@?*
	container *4
_class*
(&loc:@generator/conv2d_transpose/kernel*'
_output_shapes
:@?
?
-generator/conv2d_transpose/kernel/Adam/AssignAssign&generator/conv2d_transpose/kernel/Adam8generator/conv2d_transpose/kernel/Adam/Initializer/zeros*
T0*
use_locking(*
validate_shape(*4
_class*
(&loc:@generator/conv2d_transpose/kernel*'
_output_shapes
:@?
?
+generator/conv2d_transpose/kernel/Adam/readIdentity&generator/conv2d_transpose/kernel/Adam*
T0*4
_class*
(&loc:@generator/conv2d_transpose/kernel*'
_output_shapes
:@?
?
Jgenerator/conv2d_transpose/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"      @   ?   *
dtype0*4
_class*
(&loc:@generator/conv2d_transpose/kernel*
_output_shapes
:
?
@generator/conv2d_transpose/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*4
_class*
(&loc:@generator/conv2d_transpose/kernel*
_output_shapes
: 
?
:generator/conv2d_transpose/kernel/Adam_1/Initializer/zerosFillJgenerator/conv2d_transpose/kernel/Adam_1/Initializer/zeros/shape_as_tensor@generator/conv2d_transpose/kernel/Adam_1/Initializer/zeros/Const*
T0*

index_type0*4
_class*
(&loc:@generator/conv2d_transpose/kernel*'
_output_shapes
:@?
?
(generator/conv2d_transpose/kernel/Adam_1
VariableV2*
dtype0*
shared_name *
shape:@?*
	container *4
_class*
(&loc:@generator/conv2d_transpose/kernel*'
_output_shapes
:@?
?
/generator/conv2d_transpose/kernel/Adam_1/AssignAssign(generator/conv2d_transpose/kernel/Adam_1:generator/conv2d_transpose/kernel/Adam_1/Initializer/zeros*
T0*
use_locking(*
validate_shape(*4
_class*
(&loc:@generator/conv2d_transpose/kernel*'
_output_shapes
:@?
?
-generator/conv2d_transpose/kernel/Adam_1/readIdentity(generator/conv2d_transpose/kernel/Adam_1*
T0*4
_class*
(&loc:@generator/conv2d_transpose/kernel*'
_output_shapes
:@?
?
6generator/conv2d_transpose/bias/Adam/Initializer/zerosConst*
valueB@*    *
dtype0*2
_class(
&$loc:@generator/conv2d_transpose/bias*
_output_shapes
:@
?
$generator/conv2d_transpose/bias/Adam
VariableV2*
dtype0*
shared_name *
shape:@*
	container *2
_class(
&$loc:@generator/conv2d_transpose/bias*
_output_shapes
:@
?
+generator/conv2d_transpose/bias/Adam/AssignAssign$generator/conv2d_transpose/bias/Adam6generator/conv2d_transpose/bias/Adam/Initializer/zeros*
T0*
use_locking(*
validate_shape(*2
_class(
&$loc:@generator/conv2d_transpose/bias*
_output_shapes
:@
?
)generator/conv2d_transpose/bias/Adam/readIdentity$generator/conv2d_transpose/bias/Adam*
T0*2
_class(
&$loc:@generator/conv2d_transpose/bias*
_output_shapes
:@
?
8generator/conv2d_transpose/bias/Adam_1/Initializer/zerosConst*
valueB@*    *
dtype0*2
_class(
&$loc:@generator/conv2d_transpose/bias*
_output_shapes
:@
?
&generator/conv2d_transpose/bias/Adam_1
VariableV2*
dtype0*
shared_name *
shape:@*
	container *2
_class(
&$loc:@generator/conv2d_transpose/bias*
_output_shapes
:@
?
-generator/conv2d_transpose/bias/Adam_1/AssignAssign&generator/conv2d_transpose/bias/Adam_18generator/conv2d_transpose/bias/Adam_1/Initializer/zeros*
T0*
use_locking(*
validate_shape(*2
_class(
&$loc:@generator/conv2d_transpose/bias*
_output_shapes
:@
?
+generator/conv2d_transpose/bias/Adam_1/readIdentity&generator/conv2d_transpose/bias/Adam_1*
T0*2
_class(
&$loc:@generator/conv2d_transpose/bias*
_output_shapes
:@
?
<generator/batch_normalization_1/gamma/Adam/Initializer/zerosConst*
valueB@*    *
dtype0*8
_class.
,*loc:@generator/batch_normalization_1/gamma*
_output_shapes
:@
?
*generator/batch_normalization_1/gamma/Adam
VariableV2*
dtype0*
shared_name *
shape:@*
	container *8
_class.
,*loc:@generator/batch_normalization_1/gamma*
_output_shapes
:@
?
1generator/batch_normalization_1/gamma/Adam/AssignAssign*generator/batch_normalization_1/gamma/Adam<generator/batch_normalization_1/gamma/Adam/Initializer/zeros*
T0*
use_locking(*
validate_shape(*8
_class.
,*loc:@generator/batch_normalization_1/gamma*
_output_shapes
:@
?
/generator/batch_normalization_1/gamma/Adam/readIdentity*generator/batch_normalization_1/gamma/Adam*
T0*8
_class.
,*loc:@generator/batch_normalization_1/gamma*
_output_shapes
:@
?
>generator/batch_normalization_1/gamma/Adam_1/Initializer/zerosConst*
valueB@*    *
dtype0*8
_class.
,*loc:@generator/batch_normalization_1/gamma*
_output_shapes
:@
?
,generator/batch_normalization_1/gamma/Adam_1
VariableV2*
dtype0*
shared_name *
shape:@*
	container *8
_class.
,*loc:@generator/batch_normalization_1/gamma*
_output_shapes
:@
?
3generator/batch_normalization_1/gamma/Adam_1/AssignAssign,generator/batch_normalization_1/gamma/Adam_1>generator/batch_normalization_1/gamma/Adam_1/Initializer/zeros*
T0*
use_locking(*
validate_shape(*8
_class.
,*loc:@generator/batch_normalization_1/gamma*
_output_shapes
:@
?
1generator/batch_normalization_1/gamma/Adam_1/readIdentity,generator/batch_normalization_1/gamma/Adam_1*
T0*8
_class.
,*loc:@generator/batch_normalization_1/gamma*
_output_shapes
:@
?
;generator/batch_normalization_1/beta/Adam/Initializer/zerosConst*
valueB@*    *
dtype0*7
_class-
+)loc:@generator/batch_normalization_1/beta*
_output_shapes
:@
?
)generator/batch_normalization_1/beta/Adam
VariableV2*
dtype0*
shared_name *
shape:@*
	container *7
_class-
+)loc:@generator/batch_normalization_1/beta*
_output_shapes
:@
?
0generator/batch_normalization_1/beta/Adam/AssignAssign)generator/batch_normalization_1/beta/Adam;generator/batch_normalization_1/beta/Adam/Initializer/zeros*
T0*
use_locking(*
validate_shape(*7
_class-
+)loc:@generator/batch_normalization_1/beta*
_output_shapes
:@
?
.generator/batch_normalization_1/beta/Adam/readIdentity)generator/batch_normalization_1/beta/Adam*
T0*7
_class-
+)loc:@generator/batch_normalization_1/beta*
_output_shapes
:@
?
=generator/batch_normalization_1/beta/Adam_1/Initializer/zerosConst*
valueB@*    *
dtype0*7
_class-
+)loc:@generator/batch_normalization_1/beta*
_output_shapes
:@
?
+generator/batch_normalization_1/beta/Adam_1
VariableV2*
dtype0*
shared_name *
shape:@*
	container *7
_class-
+)loc:@generator/batch_normalization_1/beta*
_output_shapes
:@
?
2generator/batch_normalization_1/beta/Adam_1/AssignAssign+generator/batch_normalization_1/beta/Adam_1=generator/batch_normalization_1/beta/Adam_1/Initializer/zeros*
T0*
use_locking(*
validate_shape(*7
_class-
+)loc:@generator/batch_normalization_1/beta*
_output_shapes
:@
?
0generator/batch_normalization_1/beta/Adam_1/readIdentity+generator/batch_normalization_1/beta/Adam_1*
T0*7
_class-
+)loc:@generator/batch_normalization_1/beta*
_output_shapes
:@
?
Jgenerator/conv2d_transpose_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"          @   *
dtype0*6
_class,
*(loc:@generator/conv2d_transpose_1/kernel*
_output_shapes
:
?
@generator/conv2d_transpose_1/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*6
_class,
*(loc:@generator/conv2d_transpose_1/kernel*
_output_shapes
: 
?
:generator/conv2d_transpose_1/kernel/Adam/Initializer/zerosFillJgenerator/conv2d_transpose_1/kernel/Adam/Initializer/zeros/shape_as_tensor@generator/conv2d_transpose_1/kernel/Adam/Initializer/zeros/Const*
T0*

index_type0*6
_class,
*(loc:@generator/conv2d_transpose_1/kernel*&
_output_shapes
: @
?
(generator/conv2d_transpose_1/kernel/Adam
VariableV2*
dtype0*
shared_name *
shape: @*
	container *6
_class,
*(loc:@generator/conv2d_transpose_1/kernel*&
_output_shapes
: @
?
/generator/conv2d_transpose_1/kernel/Adam/AssignAssign(generator/conv2d_transpose_1/kernel/Adam:generator/conv2d_transpose_1/kernel/Adam/Initializer/zeros*
T0*
use_locking(*
validate_shape(*6
_class,
*(loc:@generator/conv2d_transpose_1/kernel*&
_output_shapes
: @
?
-generator/conv2d_transpose_1/kernel/Adam/readIdentity(generator/conv2d_transpose_1/kernel/Adam*
T0*6
_class,
*(loc:@generator/conv2d_transpose_1/kernel*&
_output_shapes
: @
?
Lgenerator/conv2d_transpose_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"          @   *
dtype0*6
_class,
*(loc:@generator/conv2d_transpose_1/kernel*
_output_shapes
:
?
Bgenerator/conv2d_transpose_1/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*6
_class,
*(loc:@generator/conv2d_transpose_1/kernel*
_output_shapes
: 
?
<generator/conv2d_transpose_1/kernel/Adam_1/Initializer/zerosFillLgenerator/conv2d_transpose_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorBgenerator/conv2d_transpose_1/kernel/Adam_1/Initializer/zeros/Const*
T0*

index_type0*6
_class,
*(loc:@generator/conv2d_transpose_1/kernel*&
_output_shapes
: @
?
*generator/conv2d_transpose_1/kernel/Adam_1
VariableV2*
dtype0*
shared_name *
shape: @*
	container *6
_class,
*(loc:@generator/conv2d_transpose_1/kernel*&
_output_shapes
: @
?
1generator/conv2d_transpose_1/kernel/Adam_1/AssignAssign*generator/conv2d_transpose_1/kernel/Adam_1<generator/conv2d_transpose_1/kernel/Adam_1/Initializer/zeros*
T0*
use_locking(*
validate_shape(*6
_class,
*(loc:@generator/conv2d_transpose_1/kernel*&
_output_shapes
: @
?
/generator/conv2d_transpose_1/kernel/Adam_1/readIdentity*generator/conv2d_transpose_1/kernel/Adam_1*
T0*6
_class,
*(loc:@generator/conv2d_transpose_1/kernel*&
_output_shapes
: @
?
8generator/conv2d_transpose_1/bias/Adam/Initializer/zerosConst*
valueB *    *
dtype0*4
_class*
(&loc:@generator/conv2d_transpose_1/bias*
_output_shapes
: 
?
&generator/conv2d_transpose_1/bias/Adam
VariableV2*
dtype0*
shared_name *
shape: *
	container *4
_class*
(&loc:@generator/conv2d_transpose_1/bias*
_output_shapes
: 
?
-generator/conv2d_transpose_1/bias/Adam/AssignAssign&generator/conv2d_transpose_1/bias/Adam8generator/conv2d_transpose_1/bias/Adam/Initializer/zeros*
T0*
use_locking(*
validate_shape(*4
_class*
(&loc:@generator/conv2d_transpose_1/bias*
_output_shapes
: 
?
+generator/conv2d_transpose_1/bias/Adam/readIdentity&generator/conv2d_transpose_1/bias/Adam*
T0*4
_class*
(&loc:@generator/conv2d_transpose_1/bias*
_output_shapes
: 
?
:generator/conv2d_transpose_1/bias/Adam_1/Initializer/zerosConst*
valueB *    *
dtype0*4
_class*
(&loc:@generator/conv2d_transpose_1/bias*
_output_shapes
: 
?
(generator/conv2d_transpose_1/bias/Adam_1
VariableV2*
dtype0*
shared_name *
shape: *
	container *4
_class*
(&loc:@generator/conv2d_transpose_1/bias*
_output_shapes
: 
?
/generator/conv2d_transpose_1/bias/Adam_1/AssignAssign(generator/conv2d_transpose_1/bias/Adam_1:generator/conv2d_transpose_1/bias/Adam_1/Initializer/zeros*
T0*
use_locking(*
validate_shape(*4
_class*
(&loc:@generator/conv2d_transpose_1/bias*
_output_shapes
: 
?
-generator/conv2d_transpose_1/bias/Adam_1/readIdentity(generator/conv2d_transpose_1/bias/Adam_1*
T0*4
_class*
(&loc:@generator/conv2d_transpose_1/bias*
_output_shapes
: 
?
<generator/batch_normalization_2/gamma/Adam/Initializer/zerosConst*
valueB *    *
dtype0*8
_class.
,*loc:@generator/batch_normalization_2/gamma*
_output_shapes
: 
?
*generator/batch_normalization_2/gamma/Adam
VariableV2*
dtype0*
shared_name *
shape: *
	container *8
_class.
,*loc:@generator/batch_normalization_2/gamma*
_output_shapes
: 
?
1generator/batch_normalization_2/gamma/Adam/AssignAssign*generator/batch_normalization_2/gamma/Adam<generator/batch_normalization_2/gamma/Adam/Initializer/zeros*
T0*
use_locking(*
validate_shape(*8
_class.
,*loc:@generator/batch_normalization_2/gamma*
_output_shapes
: 
?
/generator/batch_normalization_2/gamma/Adam/readIdentity*generator/batch_normalization_2/gamma/Adam*
T0*8
_class.
,*loc:@generator/batch_normalization_2/gamma*
_output_shapes
: 
?
>generator/batch_normalization_2/gamma/Adam_1/Initializer/zerosConst*
valueB *    *
dtype0*8
_class.
,*loc:@generator/batch_normalization_2/gamma*
_output_shapes
: 
?
,generator/batch_normalization_2/gamma/Adam_1
VariableV2*
dtype0*
shared_name *
shape: *
	container *8
_class.
,*loc:@generator/batch_normalization_2/gamma*
_output_shapes
: 
?
3generator/batch_normalization_2/gamma/Adam_1/AssignAssign,generator/batch_normalization_2/gamma/Adam_1>generator/batch_normalization_2/gamma/Adam_1/Initializer/zeros*
T0*
use_locking(*
validate_shape(*8
_class.
,*loc:@generator/batch_normalization_2/gamma*
_output_shapes
: 
?
1generator/batch_normalization_2/gamma/Adam_1/readIdentity,generator/batch_normalization_2/gamma/Adam_1*
T0*8
_class.
,*loc:@generator/batch_normalization_2/gamma*
_output_shapes
: 
?
;generator/batch_normalization_2/beta/Adam/Initializer/zerosConst*
valueB *    *
dtype0*7
_class-
+)loc:@generator/batch_normalization_2/beta*
_output_shapes
: 
?
)generator/batch_normalization_2/beta/Adam
VariableV2*
dtype0*
shared_name *
shape: *
	container *7
_class-
+)loc:@generator/batch_normalization_2/beta*
_output_shapes
: 
?
0generator/batch_normalization_2/beta/Adam/AssignAssign)generator/batch_normalization_2/beta/Adam;generator/batch_normalization_2/beta/Adam/Initializer/zeros*
T0*
use_locking(*
validate_shape(*7
_class-
+)loc:@generator/batch_normalization_2/beta*
_output_shapes
: 
?
.generator/batch_normalization_2/beta/Adam/readIdentity)generator/batch_normalization_2/beta/Adam*
T0*7
_class-
+)loc:@generator/batch_normalization_2/beta*
_output_shapes
: 
?
=generator/batch_normalization_2/beta/Adam_1/Initializer/zerosConst*
valueB *    *
dtype0*7
_class-
+)loc:@generator/batch_normalization_2/beta*
_output_shapes
: 
?
+generator/batch_normalization_2/beta/Adam_1
VariableV2*
dtype0*
shared_name *
shape: *
	container *7
_class-
+)loc:@generator/batch_normalization_2/beta*
_output_shapes
: 
?
2generator/batch_normalization_2/beta/Adam_1/AssignAssign+generator/batch_normalization_2/beta/Adam_1=generator/batch_normalization_2/beta/Adam_1/Initializer/zeros*
T0*
use_locking(*
validate_shape(*7
_class-
+)loc:@generator/batch_normalization_2/beta*
_output_shapes
: 
?
0generator/batch_normalization_2/beta/Adam_1/readIdentity+generator/batch_normalization_2/beta/Adam_1*
T0*7
_class-
+)loc:@generator/batch_normalization_2/beta*
_output_shapes
: 
?
:generator/conv2d_transpose_2/kernel/Adam/Initializer/zerosConst*%
valueB *    *
dtype0*6
_class,
*(loc:@generator/conv2d_transpose_2/kernel*&
_output_shapes
: 
?
(generator/conv2d_transpose_2/kernel/Adam
VariableV2*
dtype0*
shared_name *
shape: *
	container *6
_class,
*(loc:@generator/conv2d_transpose_2/kernel*&
_output_shapes
: 
?
/generator/conv2d_transpose_2/kernel/Adam/AssignAssign(generator/conv2d_transpose_2/kernel/Adam:generator/conv2d_transpose_2/kernel/Adam/Initializer/zeros*
T0*
use_locking(*
validate_shape(*6
_class,
*(loc:@generator/conv2d_transpose_2/kernel*&
_output_shapes
: 
?
-generator/conv2d_transpose_2/kernel/Adam/readIdentity(generator/conv2d_transpose_2/kernel/Adam*
T0*6
_class,
*(loc:@generator/conv2d_transpose_2/kernel*&
_output_shapes
: 
?
<generator/conv2d_transpose_2/kernel/Adam_1/Initializer/zerosConst*%
valueB *    *
dtype0*6
_class,
*(loc:@generator/conv2d_transpose_2/kernel*&
_output_shapes
: 
?
*generator/conv2d_transpose_2/kernel/Adam_1
VariableV2*
dtype0*
shared_name *
shape: *
	container *6
_class,
*(loc:@generator/conv2d_transpose_2/kernel*&
_output_shapes
: 
?
1generator/conv2d_transpose_2/kernel/Adam_1/AssignAssign*generator/conv2d_transpose_2/kernel/Adam_1<generator/conv2d_transpose_2/kernel/Adam_1/Initializer/zeros*
T0*
use_locking(*
validate_shape(*6
_class,
*(loc:@generator/conv2d_transpose_2/kernel*&
_output_shapes
: 
?
/generator/conv2d_transpose_2/kernel/Adam_1/readIdentity*generator/conv2d_transpose_2/kernel/Adam_1*
T0*6
_class,
*(loc:@generator/conv2d_transpose_2/kernel*&
_output_shapes
: 
?
8generator/conv2d_transpose_2/bias/Adam/Initializer/zerosConst*
valueB*    *
dtype0*4
_class*
(&loc:@generator/conv2d_transpose_2/bias*
_output_shapes
:
?
&generator/conv2d_transpose_2/bias/Adam
VariableV2*
dtype0*
shared_name *
shape:*
	container *4
_class*
(&loc:@generator/conv2d_transpose_2/bias*
_output_shapes
:
?
-generator/conv2d_transpose_2/bias/Adam/AssignAssign&generator/conv2d_transpose_2/bias/Adam8generator/conv2d_transpose_2/bias/Adam/Initializer/zeros*
T0*
use_locking(*
validate_shape(*4
_class*
(&loc:@generator/conv2d_transpose_2/bias*
_output_shapes
:
?
+generator/conv2d_transpose_2/bias/Adam/readIdentity&generator/conv2d_transpose_2/bias/Adam*
T0*4
_class*
(&loc:@generator/conv2d_transpose_2/bias*
_output_shapes
:
?
:generator/conv2d_transpose_2/bias/Adam_1/Initializer/zerosConst*
valueB*    *
dtype0*4
_class*
(&loc:@generator/conv2d_transpose_2/bias*
_output_shapes
:
?
(generator/conv2d_transpose_2/bias/Adam_1
VariableV2*
dtype0*
shared_name *
shape:*
	container *4
_class*
(&loc:@generator/conv2d_transpose_2/bias*
_output_shapes
:
?
/generator/conv2d_transpose_2/bias/Adam_1/AssignAssign(generator/conv2d_transpose_2/bias/Adam_1:generator/conv2d_transpose_2/bias/Adam_1/Initializer/zeros*
T0*
use_locking(*
validate_shape(*4
_class*
(&loc:@generator/conv2d_transpose_2/bias*
_output_shapes
:
?
-generator/conv2d_transpose_2/bias/Adam_1/readIdentity(generator/conv2d_transpose_2/bias/Adam_1*
T0*4
_class*
(&loc:@generator/conv2d_transpose_2/bias*
_output_shapes
:
Q
Adam_1/beta1Const*
valueB
 *   ?*
dtype0*
_output_shapes
: 
Q
Adam_1/beta2Const*
valueB
 *w??*
dtype0*
_output_shapes
: 
S
Adam_1/epsilonConst*
valueB
 *w?+2*
dtype0*
_output_shapes
: 
?
.Adam_1/update_generator/dense/kernel/ApplyAdam	ApplyAdamgenerator/dense/kernelgenerator/dense/kernel/Adamgenerator/dense/kernel/Adam_1beta1_power_1/readbeta2_power_1/readVariable/readAdam_1/beta1Adam_1/beta2Adam_1/epsilonBgradients_1/generator/dense/MatMul_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *)
_class
loc:@generator/dense/kernel*
_output_shapes
:	d?
?
,Adam_1/update_generator/dense/bias/ApplyAdam	ApplyAdamgenerator/dense/biasgenerator/dense/bias/Adamgenerator/dense/bias/Adam_1beta1_power_1/readbeta2_power_1/readVariable/readAdam_1/beta1Adam_1/beta2Adam_1/epsilonCgradients_1/generator/dense/BiasAdd_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *'
_class
loc:@generator/dense/bias*
_output_shapes	
:?
?
;Adam_1/update_generator/batch_normalization/gamma/ApplyAdam	ApplyAdam#generator/batch_normalization/gamma(generator/batch_normalization/gamma/Adam*generator/batch_normalization/gamma/Adam_1beta1_power_1/readbeta2_power_1/readVariable/readAdam_1/beta1Adam_1/beta2Adam_1/epsilonXgradients_1/generator/batch_normalization/FusedBatchNorm_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *6
_class,
*(loc:@generator/batch_normalization/gamma*
_output_shapes	
:?
?
:Adam_1/update_generator/batch_normalization/beta/ApplyAdam	ApplyAdam"generator/batch_normalization/beta'generator/batch_normalization/beta/Adam)generator/batch_normalization/beta/Adam_1beta1_power_1/readbeta2_power_1/readVariable/readAdam_1/beta1Adam_1/beta2Adam_1/epsilonXgradients_1/generator/batch_normalization/FusedBatchNorm_grad/tuple/control_dependency_2*
T0*
use_locking( *
use_nesterov( *5
_class+
)'loc:@generator/batch_normalization/beta*
_output_shapes	
:?
?
9Adam_1/update_generator/conv2d_transpose/kernel/ApplyAdam	ApplyAdam!generator/conv2d_transpose/kernel&generator/conv2d_transpose/kernel/Adam(generator/conv2d_transpose/kernel/Adam_1beta1_power_1/readbeta2_power_1/readVariable/readAdam_1/beta1Adam_1/beta2Adam_1/epsilonUgradients_1/generator/conv2d_transpose/conv2d_transpose_grad/tuple/control_dependency*
T0*
use_locking( *
use_nesterov( *4
_class*
(&loc:@generator/conv2d_transpose/kernel*'
_output_shapes
:@?
?
7Adam_1/update_generator/conv2d_transpose/bias/ApplyAdam	ApplyAdamgenerator/conv2d_transpose/bias$generator/conv2d_transpose/bias/Adam&generator/conv2d_transpose/bias/Adam_1beta1_power_1/readbeta2_power_1/readVariable/readAdam_1/beta1Adam_1/beta2Adam_1/epsilonNgradients_1/generator/conv2d_transpose/BiasAdd_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *2
_class(
&$loc:@generator/conv2d_transpose/bias*
_output_shapes
:@
?
=Adam_1/update_generator/batch_normalization_1/gamma/ApplyAdam	ApplyAdam%generator/batch_normalization_1/gamma*generator/batch_normalization_1/gamma/Adam,generator/batch_normalization_1/gamma/Adam_1beta1_power_1/readbeta2_power_1/readVariable/readAdam_1/beta1Adam_1/beta2Adam_1/epsilonZgradients_1/generator/batch_normalization_1/FusedBatchNorm_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *8
_class.
,*loc:@generator/batch_normalization_1/gamma*
_output_shapes
:@
?
<Adam_1/update_generator/batch_normalization_1/beta/ApplyAdam	ApplyAdam$generator/batch_normalization_1/beta)generator/batch_normalization_1/beta/Adam+generator/batch_normalization_1/beta/Adam_1beta1_power_1/readbeta2_power_1/readVariable/readAdam_1/beta1Adam_1/beta2Adam_1/epsilonZgradients_1/generator/batch_normalization_1/FusedBatchNorm_grad/tuple/control_dependency_2*
T0*
use_locking( *
use_nesterov( *7
_class-
+)loc:@generator/batch_normalization_1/beta*
_output_shapes
:@
?
;Adam_1/update_generator/conv2d_transpose_1/kernel/ApplyAdam	ApplyAdam#generator/conv2d_transpose_1/kernel(generator/conv2d_transpose_1/kernel/Adam*generator/conv2d_transpose_1/kernel/Adam_1beta1_power_1/readbeta2_power_1/readVariable/readAdam_1/beta1Adam_1/beta2Adam_1/epsilonWgradients_1/generator/conv2d_transpose_1/conv2d_transpose_grad/tuple/control_dependency*
T0*
use_locking( *
use_nesterov( *6
_class,
*(loc:@generator/conv2d_transpose_1/kernel*&
_output_shapes
: @
?
9Adam_1/update_generator/conv2d_transpose_1/bias/ApplyAdam	ApplyAdam!generator/conv2d_transpose_1/bias&generator/conv2d_transpose_1/bias/Adam(generator/conv2d_transpose_1/bias/Adam_1beta1_power_1/readbeta2_power_1/readVariable/readAdam_1/beta1Adam_1/beta2Adam_1/epsilonPgradients_1/generator/conv2d_transpose_1/BiasAdd_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *4
_class*
(&loc:@generator/conv2d_transpose_1/bias*
_output_shapes
: 
?
=Adam_1/update_generator/batch_normalization_2/gamma/ApplyAdam	ApplyAdam%generator/batch_normalization_2/gamma*generator/batch_normalization_2/gamma/Adam,generator/batch_normalization_2/gamma/Adam_1beta1_power_1/readbeta2_power_1/readVariable/readAdam_1/beta1Adam_1/beta2Adam_1/epsilonZgradients_1/generator/batch_normalization_2/FusedBatchNorm_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *8
_class.
,*loc:@generator/batch_normalization_2/gamma*
_output_shapes
: 
?
<Adam_1/update_generator/batch_normalization_2/beta/ApplyAdam	ApplyAdam$generator/batch_normalization_2/beta)generator/batch_normalization_2/beta/Adam+generator/batch_normalization_2/beta/Adam_1beta1_power_1/readbeta2_power_1/readVariable/readAdam_1/beta1Adam_1/beta2Adam_1/epsilonZgradients_1/generator/batch_normalization_2/FusedBatchNorm_grad/tuple/control_dependency_2*
T0*
use_locking( *
use_nesterov( *7
_class-
+)loc:@generator/batch_normalization_2/beta*
_output_shapes
: 
?
;Adam_1/update_generator/conv2d_transpose_2/kernel/ApplyAdam	ApplyAdam#generator/conv2d_transpose_2/kernel(generator/conv2d_transpose_2/kernel/Adam*generator/conv2d_transpose_2/kernel/Adam_1beta1_power_1/readbeta2_power_1/readVariable/readAdam_1/beta1Adam_1/beta2Adam_1/epsilonWgradients_1/generator/conv2d_transpose_2/conv2d_transpose_grad/tuple/control_dependency*
T0*
use_locking( *
use_nesterov( *6
_class,
*(loc:@generator/conv2d_transpose_2/kernel*&
_output_shapes
: 
?
9Adam_1/update_generator/conv2d_transpose_2/bias/ApplyAdam	ApplyAdam!generator/conv2d_transpose_2/bias&generator/conv2d_transpose_2/bias/Adam(generator/conv2d_transpose_2/bias/Adam_1beta1_power_1/readbeta2_power_1/readVariable/readAdam_1/beta1Adam_1/beta2Adam_1/epsilonPgradients_1/generator/conv2d_transpose_2/BiasAdd_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *4
_class*
(&loc:@generator/conv2d_transpose_2/bias*
_output_shapes
:
?

Adam_1/mulMulbeta1_power_1/readAdam_1/beta1;^Adam_1/update_generator/batch_normalization/beta/ApplyAdam<^Adam_1/update_generator/batch_normalization/gamma/ApplyAdam=^Adam_1/update_generator/batch_normalization_1/beta/ApplyAdam>^Adam_1/update_generator/batch_normalization_1/gamma/ApplyAdam=^Adam_1/update_generator/batch_normalization_2/beta/ApplyAdam>^Adam_1/update_generator/batch_normalization_2/gamma/ApplyAdam8^Adam_1/update_generator/conv2d_transpose/bias/ApplyAdam:^Adam_1/update_generator/conv2d_transpose/kernel/ApplyAdam:^Adam_1/update_generator/conv2d_transpose_1/bias/ApplyAdam<^Adam_1/update_generator/conv2d_transpose_1/kernel/ApplyAdam:^Adam_1/update_generator/conv2d_transpose_2/bias/ApplyAdam<^Adam_1/update_generator/conv2d_transpose_2/kernel/ApplyAdam-^Adam_1/update_generator/dense/bias/ApplyAdam/^Adam_1/update_generator/dense/kernel/ApplyAdam*
T0*5
_class+
)'loc:@generator/batch_normalization/beta*
_output_shapes
: 
?
Adam_1/AssignAssignbeta1_power_1
Adam_1/mul*
T0*
use_locking( *
validate_shape(*5
_class+
)'loc:@generator/batch_normalization/beta*
_output_shapes
: 
?
Adam_1/mul_1Mulbeta2_power_1/readAdam_1/beta2;^Adam_1/update_generator/batch_normalization/beta/ApplyAdam<^Adam_1/update_generator/batch_normalization/gamma/ApplyAdam=^Adam_1/update_generator/batch_normalization_1/beta/ApplyAdam>^Adam_1/update_generator/batch_normalization_1/gamma/ApplyAdam=^Adam_1/update_generator/batch_normalization_2/beta/ApplyAdam>^Adam_1/update_generator/batch_normalization_2/gamma/ApplyAdam8^Adam_1/update_generator/conv2d_transpose/bias/ApplyAdam:^Adam_1/update_generator/conv2d_transpose/kernel/ApplyAdam:^Adam_1/update_generator/conv2d_transpose_1/bias/ApplyAdam<^Adam_1/update_generator/conv2d_transpose_1/kernel/ApplyAdam:^Adam_1/update_generator/conv2d_transpose_2/bias/ApplyAdam<^Adam_1/update_generator/conv2d_transpose_2/kernel/ApplyAdam-^Adam_1/update_generator/dense/bias/ApplyAdam/^Adam_1/update_generator/dense/kernel/ApplyAdam*
T0*5
_class+
)'loc:@generator/batch_normalization/beta*
_output_shapes
: 
?
Adam_1/Assign_1Assignbeta2_power_1Adam_1/mul_1*
T0*
use_locking( *
validate_shape(*5
_class+
)'loc:@generator/batch_normalization/beta*
_output_shapes
: 
?
Adam_1NoOp^Adam_1/Assign^Adam_1/Assign_1;^Adam_1/update_generator/batch_normalization/beta/ApplyAdam<^Adam_1/update_generator/batch_normalization/gamma/ApplyAdam=^Adam_1/update_generator/batch_normalization_1/beta/ApplyAdam>^Adam_1/update_generator/batch_normalization_1/gamma/ApplyAdam=^Adam_1/update_generator/batch_normalization_2/beta/ApplyAdam>^Adam_1/update_generator/batch_normalization_2/gamma/ApplyAdam8^Adam_1/update_generator/conv2d_transpose/bias/ApplyAdam:^Adam_1/update_generator/conv2d_transpose/kernel/ApplyAdam:^Adam_1/update_generator/conv2d_transpose_1/bias/ApplyAdam<^Adam_1/update_generator/conv2d_transpose_1/kernel/ApplyAdam:^Adam_1/update_generator/conv2d_transpose_2/bias/ApplyAdam<^Adam_1/update_generator/conv2d_transpose_2/kernel/ApplyAdam-^Adam_1/update_generator/dense/bias/ApplyAdam/^Adam_1/update_generator/dense/kernel/ApplyAdam
L
mul_2/yConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 
E
mul_2MulVariable/readmul_2/y*
T0*
_output_shapes
: 
?
AssignAssignVariablemul_2*
T0*
use_locking(*
validate_shape(*
_class
loc:@Variable*
_output_shapes
: 
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
?
save/SaveV2/tensor_namesConst*?
value?B?Bdiscriminator/conv2d/biasBdiscriminator/conv2d/kernelBdiscriminator/conv2d_1/biasBdiscriminator/conv2d_1/kernelBdiscriminator/conv2d_2/biasBdiscriminator/conv2d_2/kernelBdiscriminator/conv2d_3/biasBdiscriminator/conv2d_3/kernelBdiscriminator/conv2d_4/biasBdiscriminator/conv2d_4/kernelBdiscriminator/conv2d_5/biasBdiscriminator/conv2d_5/kernelBdiscriminator/dense/biasBdiscriminator/dense/kernel*
dtype0*
_output_shapes
:

save/SaveV2/shape_and_slicesConst*/
value&B$B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
?
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesdiscriminator/conv2d/biasdiscriminator/conv2d/kerneldiscriminator/conv2d_1/biasdiscriminator/conv2d_1/kerneldiscriminator/conv2d_2/biasdiscriminator/conv2d_2/kerneldiscriminator/conv2d_3/biasdiscriminator/conv2d_3/kerneldiscriminator/conv2d_4/biasdiscriminator/conv2d_4/kerneldiscriminator/conv2d_5/biasdiscriminator/conv2d_5/kerneldiscriminator/dense/biasdiscriminator/dense/kernel*
dtypes
2
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 
?
save/RestoreV2/tensor_namesConst"/device:CPU:0*?
value?B?Bdiscriminator/conv2d/biasBdiscriminator/conv2d/kernelBdiscriminator/conv2d_1/biasBdiscriminator/conv2d_1/kernelBdiscriminator/conv2d_2/biasBdiscriminator/conv2d_2/kernelBdiscriminator/conv2d_3/biasBdiscriminator/conv2d_3/kernelBdiscriminator/conv2d_4/biasBdiscriminator/conv2d_4/kernelBdiscriminator/conv2d_5/biasBdiscriminator/conv2d_5/kernelBdiscriminator/dense/biasBdiscriminator/dense/kernel*
dtype0*
_output_shapes
:
?
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*/
value&B$B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
?
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2*L
_output_shapes:
8::::::::::::::
?
save/AssignAssigndiscriminator/conv2d/biassave/RestoreV2*
T0*
use_locking(*
validate_shape(*,
_class"
 loc:@discriminator/conv2d/bias*
_output_shapes
:@
?
save/Assign_1Assigndiscriminator/conv2d/kernelsave/RestoreV2:1*
T0*
use_locking(*
validate_shape(*.
_class$
" loc:@discriminator/conv2d/kernel*&
_output_shapes
:@
?
save/Assign_2Assigndiscriminator/conv2d_1/biassave/RestoreV2:2*
T0*
use_locking(*
validate_shape(*.
_class$
" loc:@discriminator/conv2d_1/bias*
_output_shapes
:@
?
save/Assign_3Assigndiscriminator/conv2d_1/kernelsave/RestoreV2:3*
T0*
use_locking(*
validate_shape(*0
_class&
$"loc:@discriminator/conv2d_1/kernel*&
_output_shapes
:@@
?
save/Assign_4Assigndiscriminator/conv2d_2/biassave/RestoreV2:4*
T0*
use_locking(*
validate_shape(*.
_class$
" loc:@discriminator/conv2d_2/bias*
_output_shapes
:@
?
save/Assign_5Assigndiscriminator/conv2d_2/kernelsave/RestoreV2:5*
T0*
use_locking(*
validate_shape(*0
_class&
$"loc:@discriminator/conv2d_2/kernel*&
_output_shapes
:@@
?
save/Assign_6Assigndiscriminator/conv2d_3/biassave/RestoreV2:6*
T0*
use_locking(*
validate_shape(*.
_class$
" loc:@discriminator/conv2d_3/bias*
_output_shapes	
:?
?
save/Assign_7Assigndiscriminator/conv2d_3/kernelsave/RestoreV2:7*
T0*
use_locking(*
validate_shape(*0
_class&
$"loc:@discriminator/conv2d_3/kernel*'
_output_shapes
:@?
?
save/Assign_8Assigndiscriminator/conv2d_4/biassave/RestoreV2:8*
T0*
use_locking(*
validate_shape(*.
_class$
" loc:@discriminator/conv2d_4/bias*
_output_shapes	
:?
?
save/Assign_9Assigndiscriminator/conv2d_4/kernelsave/RestoreV2:9*
T0*
use_locking(*
validate_shape(*0
_class&
$"loc:@discriminator/conv2d_4/kernel*(
_output_shapes
:??
?
save/Assign_10Assigndiscriminator/conv2d_5/biassave/RestoreV2:10*
T0*
use_locking(*
validate_shape(*.
_class$
" loc:@discriminator/conv2d_5/bias*
_output_shapes	
:?
?
save/Assign_11Assigndiscriminator/conv2d_5/kernelsave/RestoreV2:11*
T0*
use_locking(*
validate_shape(*0
_class&
$"loc:@discriminator/conv2d_5/kernel*(
_output_shapes
:??
?
save/Assign_12Assigndiscriminator/dense/biassave/RestoreV2:12*
T0*
use_locking(*
validate_shape(*+
_class!
loc:@discriminator/dense/bias*
_output_shapes
:?
?
save/Assign_13Assigndiscriminator/dense/kernelsave/RestoreV2:13*
T0*
use_locking(*
validate_shape(*-
_class#
!loc:@discriminator/dense/kernel*
_output_shapes
:	??
?
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
?%
initNoOp^Variable/Assign^beta1_power/Assign^beta1_power_1/Assign^beta2_power/Assign^beta2_power_1/Assign.^discriminator/batch_normalization/beta/Assign/^discriminator/batch_normalization/gamma/Assign5^discriminator/batch_normalization/moving_mean/Assign9^discriminator/batch_normalization/moving_variance/Assign0^discriminator/batch_normalization_1/beta/Assign1^discriminator/batch_normalization_1/gamma/Assign7^discriminator/batch_normalization_1/moving_mean/Assign;^discriminator/batch_normalization_1/moving_variance/Assign0^discriminator/batch_normalization_2/beta/Assign1^discriminator/batch_normalization_2/gamma/Assign7^discriminator/batch_normalization_2/moving_mean/Assign;^discriminator/batch_normalization_2/moving_variance/Assign&^discriminator/conv2d/bias/Adam/Assign(^discriminator/conv2d/bias/Adam_1/Assign!^discriminator/conv2d/bias/Assign(^discriminator/conv2d/kernel/Adam/Assign*^discriminator/conv2d/kernel/Adam_1/Assign#^discriminator/conv2d/kernel/Assign(^discriminator/conv2d_1/bias/Adam/Assign*^discriminator/conv2d_1/bias/Adam_1/Assign#^discriminator/conv2d_1/bias/Assign*^discriminator/conv2d_1/kernel/Adam/Assign,^discriminator/conv2d_1/kernel/Adam_1/Assign%^discriminator/conv2d_1/kernel/Assign(^discriminator/conv2d_2/bias/Adam/Assign*^discriminator/conv2d_2/bias/Adam_1/Assign#^discriminator/conv2d_2/bias/Assign*^discriminator/conv2d_2/kernel/Adam/Assign,^discriminator/conv2d_2/kernel/Adam_1/Assign%^discriminator/conv2d_2/kernel/Assign(^discriminator/conv2d_3/bias/Adam/Assign*^discriminator/conv2d_3/bias/Adam_1/Assign#^discriminator/conv2d_3/bias/Assign*^discriminator/conv2d_3/kernel/Adam/Assign,^discriminator/conv2d_3/kernel/Adam_1/Assign%^discriminator/conv2d_3/kernel/Assign(^discriminator/conv2d_4/bias/Adam/Assign*^discriminator/conv2d_4/bias/Adam_1/Assign#^discriminator/conv2d_4/bias/Assign*^discriminator/conv2d_4/kernel/Adam/Assign,^discriminator/conv2d_4/kernel/Adam_1/Assign%^discriminator/conv2d_4/kernel/Assign(^discriminator/conv2d_5/bias/Adam/Assign*^discriminator/conv2d_5/bias/Adam_1/Assign#^discriminator/conv2d_5/bias/Assign*^discriminator/conv2d_5/kernel/Adam/Assign,^discriminator/conv2d_5/kernel/Adam_1/Assign%^discriminator/conv2d_5/kernel/Assign%^discriminator/dense/bias/Adam/Assign'^discriminator/dense/bias/Adam_1/Assign ^discriminator/dense/bias/Assign'^discriminator/dense/kernel/Adam/Assign)^discriminator/dense/kernel/Adam_1/Assign"^discriminator/dense/kernel/Assign/^generator/batch_normalization/beta/Adam/Assign1^generator/batch_normalization/beta/Adam_1/Assign*^generator/batch_normalization/beta/Assign0^generator/batch_normalization/gamma/Adam/Assign2^generator/batch_normalization/gamma/Adam_1/Assign+^generator/batch_normalization/gamma/Assign1^generator/batch_normalization/moving_mean/Assign5^generator/batch_normalization/moving_variance/Assign1^generator/batch_normalization_1/beta/Adam/Assign3^generator/batch_normalization_1/beta/Adam_1/Assign,^generator/batch_normalization_1/beta/Assign2^generator/batch_normalization_1/gamma/Adam/Assign4^generator/batch_normalization_1/gamma/Adam_1/Assign-^generator/batch_normalization_1/gamma/Assign3^generator/batch_normalization_1/moving_mean/Assign7^generator/batch_normalization_1/moving_variance/Assign1^generator/batch_normalization_2/beta/Adam/Assign3^generator/batch_normalization_2/beta/Adam_1/Assign,^generator/batch_normalization_2/beta/Assign2^generator/batch_normalization_2/gamma/Adam/Assign4^generator/batch_normalization_2/gamma/Adam_1/Assign-^generator/batch_normalization_2/gamma/Assign3^generator/batch_normalization_2/moving_mean/Assign7^generator/batch_normalization_2/moving_variance/Assign,^generator/conv2d_transpose/bias/Adam/Assign.^generator/conv2d_transpose/bias/Adam_1/Assign'^generator/conv2d_transpose/bias/Assign.^generator/conv2d_transpose/kernel/Adam/Assign0^generator/conv2d_transpose/kernel/Adam_1/Assign)^generator/conv2d_transpose/kernel/Assign.^generator/conv2d_transpose_1/bias/Adam/Assign0^generator/conv2d_transpose_1/bias/Adam_1/Assign)^generator/conv2d_transpose_1/bias/Assign0^generator/conv2d_transpose_1/kernel/Adam/Assign2^generator/conv2d_transpose_1/kernel/Adam_1/Assign+^generator/conv2d_transpose_1/kernel/Assign.^generator/conv2d_transpose_2/bias/Adam/Assign0^generator/conv2d_transpose_2/bias/Adam_1/Assign)^generator/conv2d_transpose_2/bias/Assign0^generator/conv2d_transpose_2/kernel/Adam/Assign2^generator/conv2d_transpose_2/kernel/Adam_1/Assign+^generator/conv2d_transpose_2/kernel/Assign!^generator/dense/bias/Adam/Assign#^generator/dense/bias/Adam_1/Assign^generator/dense/bias/Assign#^generator/dense/kernel/Adam/Assign%^generator/dense/kernel/Adam_1/Assign^generator/dense/kernel/Assign
R
save_1/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
?
save_1/StringJoin/inputs_1Const*<
value3B1 B+_temp_d348428aca3745cfa08799bc2480720f/part*
dtype0*
_output_shapes
: 
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
S
save_1/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
m
save_1/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
?
save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards"/device:CPU:0*
_output_shapes
: 
?
save_1/SaveV2/tensor_namesConst"/device:CPU:0*?
value?B?kBVariableBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1B&discriminator/batch_normalization/betaB'discriminator/batch_normalization/gammaB-discriminator/batch_normalization/moving_meanB1discriminator/batch_normalization/moving_varianceB(discriminator/batch_normalization_1/betaB)discriminator/batch_normalization_1/gammaB/discriminator/batch_normalization_1/moving_meanB3discriminator/batch_normalization_1/moving_varianceB(discriminator/batch_normalization_2/betaB)discriminator/batch_normalization_2/gammaB/discriminator/batch_normalization_2/moving_meanB3discriminator/batch_normalization_2/moving_varianceBdiscriminator/conv2d/biasBdiscriminator/conv2d/bias/AdamB discriminator/conv2d/bias/Adam_1Bdiscriminator/conv2d/kernelB discriminator/conv2d/kernel/AdamB"discriminator/conv2d/kernel/Adam_1Bdiscriminator/conv2d_1/biasB discriminator/conv2d_1/bias/AdamB"discriminator/conv2d_1/bias/Adam_1Bdiscriminator/conv2d_1/kernelB"discriminator/conv2d_1/kernel/AdamB$discriminator/conv2d_1/kernel/Adam_1Bdiscriminator/conv2d_2/biasB discriminator/conv2d_2/bias/AdamB"discriminator/conv2d_2/bias/Adam_1Bdiscriminator/conv2d_2/kernelB"discriminator/conv2d_2/kernel/AdamB$discriminator/conv2d_2/kernel/Adam_1Bdiscriminator/conv2d_3/biasB discriminator/conv2d_3/bias/AdamB"discriminator/conv2d_3/bias/Adam_1Bdiscriminator/conv2d_3/kernelB"discriminator/conv2d_3/kernel/AdamB$discriminator/conv2d_3/kernel/Adam_1Bdiscriminator/conv2d_4/biasB discriminator/conv2d_4/bias/AdamB"discriminator/conv2d_4/bias/Adam_1Bdiscriminator/conv2d_4/kernelB"discriminator/conv2d_4/kernel/AdamB$discriminator/conv2d_4/kernel/Adam_1Bdiscriminator/conv2d_5/biasB discriminator/conv2d_5/bias/AdamB"discriminator/conv2d_5/bias/Adam_1Bdiscriminator/conv2d_5/kernelB"discriminator/conv2d_5/kernel/AdamB$discriminator/conv2d_5/kernel/Adam_1Bdiscriminator/dense/biasBdiscriminator/dense/bias/AdamBdiscriminator/dense/bias/Adam_1Bdiscriminator/dense/kernelBdiscriminator/dense/kernel/AdamB!discriminator/dense/kernel/Adam_1B"generator/batch_normalization/betaB'generator/batch_normalization/beta/AdamB)generator/batch_normalization/beta/Adam_1B#generator/batch_normalization/gammaB(generator/batch_normalization/gamma/AdamB*generator/batch_normalization/gamma/Adam_1B)generator/batch_normalization/moving_meanB-generator/batch_normalization/moving_varianceB$generator/batch_normalization_1/betaB)generator/batch_normalization_1/beta/AdamB+generator/batch_normalization_1/beta/Adam_1B%generator/batch_normalization_1/gammaB*generator/batch_normalization_1/gamma/AdamB,generator/batch_normalization_1/gamma/Adam_1B+generator/batch_normalization_1/moving_meanB/generator/batch_normalization_1/moving_varianceB$generator/batch_normalization_2/betaB)generator/batch_normalization_2/beta/AdamB+generator/batch_normalization_2/beta/Adam_1B%generator/batch_normalization_2/gammaB*generator/batch_normalization_2/gamma/AdamB,generator/batch_normalization_2/gamma/Adam_1B+generator/batch_normalization_2/moving_meanB/generator/batch_normalization_2/moving_varianceBgenerator/conv2d_transpose/biasB$generator/conv2d_transpose/bias/AdamB&generator/conv2d_transpose/bias/Adam_1B!generator/conv2d_transpose/kernelB&generator/conv2d_transpose/kernel/AdamB(generator/conv2d_transpose/kernel/Adam_1B!generator/conv2d_transpose_1/biasB&generator/conv2d_transpose_1/bias/AdamB(generator/conv2d_transpose_1/bias/Adam_1B#generator/conv2d_transpose_1/kernelB(generator/conv2d_transpose_1/kernel/AdamB*generator/conv2d_transpose_1/kernel/Adam_1B!generator/conv2d_transpose_2/biasB&generator/conv2d_transpose_2/bias/AdamB(generator/conv2d_transpose_2/bias/Adam_1B#generator/conv2d_transpose_2/kernelB(generator/conv2d_transpose_2/kernel/AdamB*generator/conv2d_transpose_2/kernel/Adam_1Bgenerator/dense/biasBgenerator/dense/bias/AdamBgenerator/dense/bias/Adam_1Bgenerator/dense/kernelBgenerator/dense/kernel/AdamBgenerator/dense/kernel/Adam_1*
dtype0*
_output_shapes
:k
?
save_1/SaveV2/shape_and_slicesConst"/device:CPU:0*?
value?B?kB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:k
? 
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesVariablebeta1_powerbeta1_power_1beta2_powerbeta2_power_1&discriminator/batch_normalization/beta'discriminator/batch_normalization/gamma-discriminator/batch_normalization/moving_mean1discriminator/batch_normalization/moving_variance(discriminator/batch_normalization_1/beta)discriminator/batch_normalization_1/gamma/discriminator/batch_normalization_1/moving_mean3discriminator/batch_normalization_1/moving_variance(discriminator/batch_normalization_2/beta)discriminator/batch_normalization_2/gamma/discriminator/batch_normalization_2/moving_mean3discriminator/batch_normalization_2/moving_variancediscriminator/conv2d/biasdiscriminator/conv2d/bias/Adam discriminator/conv2d/bias/Adam_1discriminator/conv2d/kernel discriminator/conv2d/kernel/Adam"discriminator/conv2d/kernel/Adam_1discriminator/conv2d_1/bias discriminator/conv2d_1/bias/Adam"discriminator/conv2d_1/bias/Adam_1discriminator/conv2d_1/kernel"discriminator/conv2d_1/kernel/Adam$discriminator/conv2d_1/kernel/Adam_1discriminator/conv2d_2/bias discriminator/conv2d_2/bias/Adam"discriminator/conv2d_2/bias/Adam_1discriminator/conv2d_2/kernel"discriminator/conv2d_2/kernel/Adam$discriminator/conv2d_2/kernel/Adam_1discriminator/conv2d_3/bias discriminator/conv2d_3/bias/Adam"discriminator/conv2d_3/bias/Adam_1discriminator/conv2d_3/kernel"discriminator/conv2d_3/kernel/Adam$discriminator/conv2d_3/kernel/Adam_1discriminator/conv2d_4/bias discriminator/conv2d_4/bias/Adam"discriminator/conv2d_4/bias/Adam_1discriminator/conv2d_4/kernel"discriminator/conv2d_4/kernel/Adam$discriminator/conv2d_4/kernel/Adam_1discriminator/conv2d_5/bias discriminator/conv2d_5/bias/Adam"discriminator/conv2d_5/bias/Adam_1discriminator/conv2d_5/kernel"discriminator/conv2d_5/kernel/Adam$discriminator/conv2d_5/kernel/Adam_1discriminator/dense/biasdiscriminator/dense/bias/Adamdiscriminator/dense/bias/Adam_1discriminator/dense/kerneldiscriminator/dense/kernel/Adam!discriminator/dense/kernel/Adam_1"generator/batch_normalization/beta'generator/batch_normalization/beta/Adam)generator/batch_normalization/beta/Adam_1#generator/batch_normalization/gamma(generator/batch_normalization/gamma/Adam*generator/batch_normalization/gamma/Adam_1)generator/batch_normalization/moving_mean-generator/batch_normalization/moving_variance$generator/batch_normalization_1/beta)generator/batch_normalization_1/beta/Adam+generator/batch_normalization_1/beta/Adam_1%generator/batch_normalization_1/gamma*generator/batch_normalization_1/gamma/Adam,generator/batch_normalization_1/gamma/Adam_1+generator/batch_normalization_1/moving_mean/generator/batch_normalization_1/moving_variance$generator/batch_normalization_2/beta)generator/batch_normalization_2/beta/Adam+generator/batch_normalization_2/beta/Adam_1%generator/batch_normalization_2/gamma*generator/batch_normalization_2/gamma/Adam,generator/batch_normalization_2/gamma/Adam_1+generator/batch_normalization_2/moving_mean/generator/batch_normalization_2/moving_variancegenerator/conv2d_transpose/bias$generator/conv2d_transpose/bias/Adam&generator/conv2d_transpose/bias/Adam_1!generator/conv2d_transpose/kernel&generator/conv2d_transpose/kernel/Adam(generator/conv2d_transpose/kernel/Adam_1!generator/conv2d_transpose_1/bias&generator/conv2d_transpose_1/bias/Adam(generator/conv2d_transpose_1/bias/Adam_1#generator/conv2d_transpose_1/kernel(generator/conv2d_transpose_1/kernel/Adam*generator/conv2d_transpose_1/kernel/Adam_1!generator/conv2d_transpose_2/bias&generator/conv2d_transpose_2/bias/Adam(generator/conv2d_transpose_2/bias/Adam_1#generator/conv2d_transpose_2/kernel(generator/conv2d_transpose_2/kernel/Adam*generator/conv2d_transpose_2/kernel/Adam_1generator/dense/biasgenerator/dense/bias/Adamgenerator/dense/bias/Adam_1generator/dense/kernelgenerator/dense/kernel/Adamgenerator/dense/kernel/Adam_1"/device:CPU:0*y
dtypeso
m2k
?
save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2"/device:CPU:0*
T0*)
_class
loc:@save_1/ShardedFilename*
_output_shapes
: 
?
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency"/device:CPU:0*

axis *
T0*
N*
_output_shapes
:
?
save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const"/device:CPU:0*
delete_old_dirs(
?
save_1/IdentityIdentitysave_1/Const^save_1/MergeV2Checkpoints^save_1/control_dependency"/device:CPU:0*
T0*
_output_shapes
: 
?
save_1/RestoreV2/tensor_namesConst"/device:CPU:0*?
value?B?kBVariableBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1B&discriminator/batch_normalization/betaB'discriminator/batch_normalization/gammaB-discriminator/batch_normalization/moving_meanB1discriminator/batch_normalization/moving_varianceB(discriminator/batch_normalization_1/betaB)discriminator/batch_normalization_1/gammaB/discriminator/batch_normalization_1/moving_meanB3discriminator/batch_normalization_1/moving_varianceB(discriminator/batch_normalization_2/betaB)discriminator/batch_normalization_2/gammaB/discriminator/batch_normalization_2/moving_meanB3discriminator/batch_normalization_2/moving_varianceBdiscriminator/conv2d/biasBdiscriminator/conv2d/bias/AdamB discriminator/conv2d/bias/Adam_1Bdiscriminator/conv2d/kernelB discriminator/conv2d/kernel/AdamB"discriminator/conv2d/kernel/Adam_1Bdiscriminator/conv2d_1/biasB discriminator/conv2d_1/bias/AdamB"discriminator/conv2d_1/bias/Adam_1Bdiscriminator/conv2d_1/kernelB"discriminator/conv2d_1/kernel/AdamB$discriminator/conv2d_1/kernel/Adam_1Bdiscriminator/conv2d_2/biasB discriminator/conv2d_2/bias/AdamB"discriminator/conv2d_2/bias/Adam_1Bdiscriminator/conv2d_2/kernelB"discriminator/conv2d_2/kernel/AdamB$discriminator/conv2d_2/kernel/Adam_1Bdiscriminator/conv2d_3/biasB discriminator/conv2d_3/bias/AdamB"discriminator/conv2d_3/bias/Adam_1Bdiscriminator/conv2d_3/kernelB"discriminator/conv2d_3/kernel/AdamB$discriminator/conv2d_3/kernel/Adam_1Bdiscriminator/conv2d_4/biasB discriminator/conv2d_4/bias/AdamB"discriminator/conv2d_4/bias/Adam_1Bdiscriminator/conv2d_4/kernelB"discriminator/conv2d_4/kernel/AdamB$discriminator/conv2d_4/kernel/Adam_1Bdiscriminator/conv2d_5/biasB discriminator/conv2d_5/bias/AdamB"discriminator/conv2d_5/bias/Adam_1Bdiscriminator/conv2d_5/kernelB"discriminator/conv2d_5/kernel/AdamB$discriminator/conv2d_5/kernel/Adam_1Bdiscriminator/dense/biasBdiscriminator/dense/bias/AdamBdiscriminator/dense/bias/Adam_1Bdiscriminator/dense/kernelBdiscriminator/dense/kernel/AdamB!discriminator/dense/kernel/Adam_1B"generator/batch_normalization/betaB'generator/batch_normalization/beta/AdamB)generator/batch_normalization/beta/Adam_1B#generator/batch_normalization/gammaB(generator/batch_normalization/gamma/AdamB*generator/batch_normalization/gamma/Adam_1B)generator/batch_normalization/moving_meanB-generator/batch_normalization/moving_varianceB$generator/batch_normalization_1/betaB)generator/batch_normalization_1/beta/AdamB+generator/batch_normalization_1/beta/Adam_1B%generator/batch_normalization_1/gammaB*generator/batch_normalization_1/gamma/AdamB,generator/batch_normalization_1/gamma/Adam_1B+generator/batch_normalization_1/moving_meanB/generator/batch_normalization_1/moving_varianceB$generator/batch_normalization_2/betaB)generator/batch_normalization_2/beta/AdamB+generator/batch_normalization_2/beta/Adam_1B%generator/batch_normalization_2/gammaB*generator/batch_normalization_2/gamma/AdamB,generator/batch_normalization_2/gamma/Adam_1B+generator/batch_normalization_2/moving_meanB/generator/batch_normalization_2/moving_varianceBgenerator/conv2d_transpose/biasB$generator/conv2d_transpose/bias/AdamB&generator/conv2d_transpose/bias/Adam_1B!generator/conv2d_transpose/kernelB&generator/conv2d_transpose/kernel/AdamB(generator/conv2d_transpose/kernel/Adam_1B!generator/conv2d_transpose_1/biasB&generator/conv2d_transpose_1/bias/AdamB(generator/conv2d_transpose_1/bias/Adam_1B#generator/conv2d_transpose_1/kernelB(generator/conv2d_transpose_1/kernel/AdamB*generator/conv2d_transpose_1/kernel/Adam_1B!generator/conv2d_transpose_2/biasB&generator/conv2d_transpose_2/bias/AdamB(generator/conv2d_transpose_2/bias/Adam_1B#generator/conv2d_transpose_2/kernelB(generator/conv2d_transpose_2/kernel/AdamB*generator/conv2d_transpose_2/kernel/Adam_1Bgenerator/dense/biasBgenerator/dense/bias/AdamBgenerator/dense/bias/Adam_1Bgenerator/dense/kernelBgenerator/dense/kernel/AdamBgenerator/dense/kernel/Adam_1*
dtype0*
_output_shapes
:k
?
!save_1/RestoreV2/shape_and_slicesConst"/device:CPU:0*?
value?B?kB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:k
?
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices"/device:CPU:0*y
dtypeso
m2k*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
?
save_1/AssignAssignVariablesave_1/RestoreV2*
T0*
use_locking(*
validate_shape(*
_class
loc:@Variable*
_output_shapes
: 
?
save_1/Assign_1Assignbeta1_powersave_1/RestoreV2:1*
T0*
use_locking(*
validate_shape(*,
_class"
 loc:@discriminator/conv2d/bias*
_output_shapes
: 
?
save_1/Assign_2Assignbeta1_power_1save_1/RestoreV2:2*
T0*
use_locking(*
validate_shape(*5
_class+
)'loc:@generator/batch_normalization/beta*
_output_shapes
: 
?
save_1/Assign_3Assignbeta2_powersave_1/RestoreV2:3*
T0*
use_locking(*
validate_shape(*,
_class"
 loc:@discriminator/conv2d/bias*
_output_shapes
: 
?
save_1/Assign_4Assignbeta2_power_1save_1/RestoreV2:4*
T0*
use_locking(*
validate_shape(*5
_class+
)'loc:@generator/batch_normalization/beta*
_output_shapes
: 
?
save_1/Assign_5Assign&discriminator/batch_normalization/betasave_1/RestoreV2:5*
T0*
use_locking(*
validate_shape(*9
_class/
-+loc:@discriminator/batch_normalization/beta*
_output_shapes
:@
?
save_1/Assign_6Assign'discriminator/batch_normalization/gammasave_1/RestoreV2:6*
T0*
use_locking(*
validate_shape(*:
_class0
.,loc:@discriminator/batch_normalization/gamma*
_output_shapes
:@
?
save_1/Assign_7Assign-discriminator/batch_normalization/moving_meansave_1/RestoreV2:7*
T0*
use_locking(*
validate_shape(*@
_class6
42loc:@discriminator/batch_normalization/moving_mean*
_output_shapes
:@
?
save_1/Assign_8Assign1discriminator/batch_normalization/moving_variancesave_1/RestoreV2:8*
T0*
use_locking(*
validate_shape(*D
_class:
86loc:@discriminator/batch_normalization/moving_variance*
_output_shapes
:@
?
save_1/Assign_9Assign(discriminator/batch_normalization_1/betasave_1/RestoreV2:9*
T0*
use_locking(*
validate_shape(*;
_class1
/-loc:@discriminator/batch_normalization_1/beta*
_output_shapes	
:?
?
save_1/Assign_10Assign)discriminator/batch_normalization_1/gammasave_1/RestoreV2:10*
T0*
use_locking(*
validate_shape(*<
_class2
0.loc:@discriminator/batch_normalization_1/gamma*
_output_shapes	
:?
?
save_1/Assign_11Assign/discriminator/batch_normalization_1/moving_meansave_1/RestoreV2:11*
T0*
use_locking(*
validate_shape(*B
_class8
64loc:@discriminator/batch_normalization_1/moving_mean*
_output_shapes	
:?
?
save_1/Assign_12Assign3discriminator/batch_normalization_1/moving_variancesave_1/RestoreV2:12*
T0*
use_locking(*
validate_shape(*F
_class<
:8loc:@discriminator/batch_normalization_1/moving_variance*
_output_shapes	
:?
?
save_1/Assign_13Assign(discriminator/batch_normalization_2/betasave_1/RestoreV2:13*
T0*
use_locking(*
validate_shape(*;
_class1
/-loc:@discriminator/batch_normalization_2/beta*
_output_shapes	
:?
?
save_1/Assign_14Assign)discriminator/batch_normalization_2/gammasave_1/RestoreV2:14*
T0*
use_locking(*
validate_shape(*<
_class2
0.loc:@discriminator/batch_normalization_2/gamma*
_output_shapes	
:?
?
save_1/Assign_15Assign/discriminator/batch_normalization_2/moving_meansave_1/RestoreV2:15*
T0*
use_locking(*
validate_shape(*B
_class8
64loc:@discriminator/batch_normalization_2/moving_mean*
_output_shapes	
:?
?
save_1/Assign_16Assign3discriminator/batch_normalization_2/moving_variancesave_1/RestoreV2:16*
T0*
use_locking(*
validate_shape(*F
_class<
:8loc:@discriminator/batch_normalization_2/moving_variance*
_output_shapes	
:?
?
save_1/Assign_17Assigndiscriminator/conv2d/biassave_1/RestoreV2:17*
T0*
use_locking(*
validate_shape(*,
_class"
 loc:@discriminator/conv2d/bias*
_output_shapes
:@
?
save_1/Assign_18Assigndiscriminator/conv2d/bias/Adamsave_1/RestoreV2:18*
T0*
use_locking(*
validate_shape(*,
_class"
 loc:@discriminator/conv2d/bias*
_output_shapes
:@
?
save_1/Assign_19Assign discriminator/conv2d/bias/Adam_1save_1/RestoreV2:19*
T0*
use_locking(*
validate_shape(*,
_class"
 loc:@discriminator/conv2d/bias*
_output_shapes
:@
?
save_1/Assign_20Assigndiscriminator/conv2d/kernelsave_1/RestoreV2:20*
T0*
use_locking(*
validate_shape(*.
_class$
" loc:@discriminator/conv2d/kernel*&
_output_shapes
:@
?
save_1/Assign_21Assign discriminator/conv2d/kernel/Adamsave_1/RestoreV2:21*
T0*
use_locking(*
validate_shape(*.
_class$
" loc:@discriminator/conv2d/kernel*&
_output_shapes
:@
?
save_1/Assign_22Assign"discriminator/conv2d/kernel/Adam_1save_1/RestoreV2:22*
T0*
use_locking(*
validate_shape(*.
_class$
" loc:@discriminator/conv2d/kernel*&
_output_shapes
:@
?
save_1/Assign_23Assigndiscriminator/conv2d_1/biassave_1/RestoreV2:23*
T0*
use_locking(*
validate_shape(*.
_class$
" loc:@discriminator/conv2d_1/bias*
_output_shapes
:@
?
save_1/Assign_24Assign discriminator/conv2d_1/bias/Adamsave_1/RestoreV2:24*
T0*
use_locking(*
validate_shape(*.
_class$
" loc:@discriminator/conv2d_1/bias*
_output_shapes
:@
?
save_1/Assign_25Assign"discriminator/conv2d_1/bias/Adam_1save_1/RestoreV2:25*
T0*
use_locking(*
validate_shape(*.
_class$
" loc:@discriminator/conv2d_1/bias*
_output_shapes
:@
?
save_1/Assign_26Assigndiscriminator/conv2d_1/kernelsave_1/RestoreV2:26*
T0*
use_locking(*
validate_shape(*0
_class&
$"loc:@discriminator/conv2d_1/kernel*&
_output_shapes
:@@
?
save_1/Assign_27Assign"discriminator/conv2d_1/kernel/Adamsave_1/RestoreV2:27*
T0*
use_locking(*
validate_shape(*0
_class&
$"loc:@discriminator/conv2d_1/kernel*&
_output_shapes
:@@
?
save_1/Assign_28Assign$discriminator/conv2d_1/kernel/Adam_1save_1/RestoreV2:28*
T0*
use_locking(*
validate_shape(*0
_class&
$"loc:@discriminator/conv2d_1/kernel*&
_output_shapes
:@@
?
save_1/Assign_29Assigndiscriminator/conv2d_2/biassave_1/RestoreV2:29*
T0*
use_locking(*
validate_shape(*.
_class$
" loc:@discriminator/conv2d_2/bias*
_output_shapes
:@
?
save_1/Assign_30Assign discriminator/conv2d_2/bias/Adamsave_1/RestoreV2:30*
T0*
use_locking(*
validate_shape(*.
_class$
" loc:@discriminator/conv2d_2/bias*
_output_shapes
:@
?
save_1/Assign_31Assign"discriminator/conv2d_2/bias/Adam_1save_1/RestoreV2:31*
T0*
use_locking(*
validate_shape(*.
_class$
" loc:@discriminator/conv2d_2/bias*
_output_shapes
:@
?
save_1/Assign_32Assigndiscriminator/conv2d_2/kernelsave_1/RestoreV2:32*
T0*
use_locking(*
validate_shape(*0
_class&
$"loc:@discriminator/conv2d_2/kernel*&
_output_shapes
:@@
?
save_1/Assign_33Assign"discriminator/conv2d_2/kernel/Adamsave_1/RestoreV2:33*
T0*
use_locking(*
validate_shape(*0
_class&
$"loc:@discriminator/conv2d_2/kernel*&
_output_shapes
:@@
?
save_1/Assign_34Assign$discriminator/conv2d_2/kernel/Adam_1save_1/RestoreV2:34*
T0*
use_locking(*
validate_shape(*0
_class&
$"loc:@discriminator/conv2d_2/kernel*&
_output_shapes
:@@
?
save_1/Assign_35Assigndiscriminator/conv2d_3/biassave_1/RestoreV2:35*
T0*
use_locking(*
validate_shape(*.
_class$
" loc:@discriminator/conv2d_3/bias*
_output_shapes	
:?
?
save_1/Assign_36Assign discriminator/conv2d_3/bias/Adamsave_1/RestoreV2:36*
T0*
use_locking(*
validate_shape(*.
_class$
" loc:@discriminator/conv2d_3/bias*
_output_shapes	
:?
?
save_1/Assign_37Assign"discriminator/conv2d_3/bias/Adam_1save_1/RestoreV2:37*
T0*
use_locking(*
validate_shape(*.
_class$
" loc:@discriminator/conv2d_3/bias*
_output_shapes	
:?
?
save_1/Assign_38Assigndiscriminator/conv2d_3/kernelsave_1/RestoreV2:38*
T0*
use_locking(*
validate_shape(*0
_class&
$"loc:@discriminator/conv2d_3/kernel*'
_output_shapes
:@?
?
save_1/Assign_39Assign"discriminator/conv2d_3/kernel/Adamsave_1/RestoreV2:39*
T0*
use_locking(*
validate_shape(*0
_class&
$"loc:@discriminator/conv2d_3/kernel*'
_output_shapes
:@?
?
save_1/Assign_40Assign$discriminator/conv2d_3/kernel/Adam_1save_1/RestoreV2:40*
T0*
use_locking(*
validate_shape(*0
_class&
$"loc:@discriminator/conv2d_3/kernel*'
_output_shapes
:@?
?
save_1/Assign_41Assigndiscriminator/conv2d_4/biassave_1/RestoreV2:41*
T0*
use_locking(*
validate_shape(*.
_class$
" loc:@discriminator/conv2d_4/bias*
_output_shapes	
:?
?
save_1/Assign_42Assign discriminator/conv2d_4/bias/Adamsave_1/RestoreV2:42*
T0*
use_locking(*
validate_shape(*.
_class$
" loc:@discriminator/conv2d_4/bias*
_output_shapes	
:?
?
save_1/Assign_43Assign"discriminator/conv2d_4/bias/Adam_1save_1/RestoreV2:43*
T0*
use_locking(*
validate_shape(*.
_class$
" loc:@discriminator/conv2d_4/bias*
_output_shapes	
:?
?
save_1/Assign_44Assigndiscriminator/conv2d_4/kernelsave_1/RestoreV2:44*
T0*
use_locking(*
validate_shape(*0
_class&
$"loc:@discriminator/conv2d_4/kernel*(
_output_shapes
:??
?
save_1/Assign_45Assign"discriminator/conv2d_4/kernel/Adamsave_1/RestoreV2:45*
T0*
use_locking(*
validate_shape(*0
_class&
$"loc:@discriminator/conv2d_4/kernel*(
_output_shapes
:??
?
save_1/Assign_46Assign$discriminator/conv2d_4/kernel/Adam_1save_1/RestoreV2:46*
T0*
use_locking(*
validate_shape(*0
_class&
$"loc:@discriminator/conv2d_4/kernel*(
_output_shapes
:??
?
save_1/Assign_47Assigndiscriminator/conv2d_5/biassave_1/RestoreV2:47*
T0*
use_locking(*
validate_shape(*.
_class$
" loc:@discriminator/conv2d_5/bias*
_output_shapes	
:?
?
save_1/Assign_48Assign discriminator/conv2d_5/bias/Adamsave_1/RestoreV2:48*
T0*
use_locking(*
validate_shape(*.
_class$
" loc:@discriminator/conv2d_5/bias*
_output_shapes	
:?
?
save_1/Assign_49Assign"discriminator/conv2d_5/bias/Adam_1save_1/RestoreV2:49*
T0*
use_locking(*
validate_shape(*.
_class$
" loc:@discriminator/conv2d_5/bias*
_output_shapes	
:?
?
save_1/Assign_50Assigndiscriminator/conv2d_5/kernelsave_1/RestoreV2:50*
T0*
use_locking(*
validate_shape(*0
_class&
$"loc:@discriminator/conv2d_5/kernel*(
_output_shapes
:??
?
save_1/Assign_51Assign"discriminator/conv2d_5/kernel/Adamsave_1/RestoreV2:51*
T0*
use_locking(*
validate_shape(*0
_class&
$"loc:@discriminator/conv2d_5/kernel*(
_output_shapes
:??
?
save_1/Assign_52Assign$discriminator/conv2d_5/kernel/Adam_1save_1/RestoreV2:52*
T0*
use_locking(*
validate_shape(*0
_class&
$"loc:@discriminator/conv2d_5/kernel*(
_output_shapes
:??
?
save_1/Assign_53Assigndiscriminator/dense/biassave_1/RestoreV2:53*
T0*
use_locking(*
validate_shape(*+
_class!
loc:@discriminator/dense/bias*
_output_shapes
:?
?
save_1/Assign_54Assigndiscriminator/dense/bias/Adamsave_1/RestoreV2:54*
T0*
use_locking(*
validate_shape(*+
_class!
loc:@discriminator/dense/bias*
_output_shapes
:?
?
save_1/Assign_55Assigndiscriminator/dense/bias/Adam_1save_1/RestoreV2:55*
T0*
use_locking(*
validate_shape(*+
_class!
loc:@discriminator/dense/bias*
_output_shapes
:?
?
save_1/Assign_56Assigndiscriminator/dense/kernelsave_1/RestoreV2:56*
T0*
use_locking(*
validate_shape(*-
_class#
!loc:@discriminator/dense/kernel*
_output_shapes
:	??
?
save_1/Assign_57Assigndiscriminator/dense/kernel/Adamsave_1/RestoreV2:57*
T0*
use_locking(*
validate_shape(*-
_class#
!loc:@discriminator/dense/kernel*
_output_shapes
:	??
?
save_1/Assign_58Assign!discriminator/dense/kernel/Adam_1save_1/RestoreV2:58*
T0*
use_locking(*
validate_shape(*-
_class#
!loc:@discriminator/dense/kernel*
_output_shapes
:	??
?
save_1/Assign_59Assign"generator/batch_normalization/betasave_1/RestoreV2:59*
T0*
use_locking(*
validate_shape(*5
_class+
)'loc:@generator/batch_normalization/beta*
_output_shapes	
:?
?
save_1/Assign_60Assign'generator/batch_normalization/beta/Adamsave_1/RestoreV2:60*
T0*
use_locking(*
validate_shape(*5
_class+
)'loc:@generator/batch_normalization/beta*
_output_shapes	
:?
?
save_1/Assign_61Assign)generator/batch_normalization/beta/Adam_1save_1/RestoreV2:61*
T0*
use_locking(*
validate_shape(*5
_class+
)'loc:@generator/batch_normalization/beta*
_output_shapes	
:?
?
save_1/Assign_62Assign#generator/batch_normalization/gammasave_1/RestoreV2:62*
T0*
use_locking(*
validate_shape(*6
_class,
*(loc:@generator/batch_normalization/gamma*
_output_shapes	
:?
?
save_1/Assign_63Assign(generator/batch_normalization/gamma/Adamsave_1/RestoreV2:63*
T0*
use_locking(*
validate_shape(*6
_class,
*(loc:@generator/batch_normalization/gamma*
_output_shapes	
:?
?
save_1/Assign_64Assign*generator/batch_normalization/gamma/Adam_1save_1/RestoreV2:64*
T0*
use_locking(*
validate_shape(*6
_class,
*(loc:@generator/batch_normalization/gamma*
_output_shapes	
:?
?
save_1/Assign_65Assign)generator/batch_normalization/moving_meansave_1/RestoreV2:65*
T0*
use_locking(*
validate_shape(*<
_class2
0.loc:@generator/batch_normalization/moving_mean*
_output_shapes	
:?
?
save_1/Assign_66Assign-generator/batch_normalization/moving_variancesave_1/RestoreV2:66*
T0*
use_locking(*
validate_shape(*@
_class6
42loc:@generator/batch_normalization/moving_variance*
_output_shapes	
:?
?
save_1/Assign_67Assign$generator/batch_normalization_1/betasave_1/RestoreV2:67*
T0*
use_locking(*
validate_shape(*7
_class-
+)loc:@generator/batch_normalization_1/beta*
_output_shapes
:@
?
save_1/Assign_68Assign)generator/batch_normalization_1/beta/Adamsave_1/RestoreV2:68*
T0*
use_locking(*
validate_shape(*7
_class-
+)loc:@generator/batch_normalization_1/beta*
_output_shapes
:@
?
save_1/Assign_69Assign+generator/batch_normalization_1/beta/Adam_1save_1/RestoreV2:69*
T0*
use_locking(*
validate_shape(*7
_class-
+)loc:@generator/batch_normalization_1/beta*
_output_shapes
:@
?
save_1/Assign_70Assign%generator/batch_normalization_1/gammasave_1/RestoreV2:70*
T0*
use_locking(*
validate_shape(*8
_class.
,*loc:@generator/batch_normalization_1/gamma*
_output_shapes
:@
?
save_1/Assign_71Assign*generator/batch_normalization_1/gamma/Adamsave_1/RestoreV2:71*
T0*
use_locking(*
validate_shape(*8
_class.
,*loc:@generator/batch_normalization_1/gamma*
_output_shapes
:@
?
save_1/Assign_72Assign,generator/batch_normalization_1/gamma/Adam_1save_1/RestoreV2:72*
T0*
use_locking(*
validate_shape(*8
_class.
,*loc:@generator/batch_normalization_1/gamma*
_output_shapes
:@
?
save_1/Assign_73Assign+generator/batch_normalization_1/moving_meansave_1/RestoreV2:73*
T0*
use_locking(*
validate_shape(*>
_class4
20loc:@generator/batch_normalization_1/moving_mean*
_output_shapes
:@
?
save_1/Assign_74Assign/generator/batch_normalization_1/moving_variancesave_1/RestoreV2:74*
T0*
use_locking(*
validate_shape(*B
_class8
64loc:@generator/batch_normalization_1/moving_variance*
_output_shapes
:@
?
save_1/Assign_75Assign$generator/batch_normalization_2/betasave_1/RestoreV2:75*
T0*
use_locking(*
validate_shape(*7
_class-
+)loc:@generator/batch_normalization_2/beta*
_output_shapes
: 
?
save_1/Assign_76Assign)generator/batch_normalization_2/beta/Adamsave_1/RestoreV2:76*
T0*
use_locking(*
validate_shape(*7
_class-
+)loc:@generator/batch_normalization_2/beta*
_output_shapes
: 
?
save_1/Assign_77Assign+generator/batch_normalization_2/beta/Adam_1save_1/RestoreV2:77*
T0*
use_locking(*
validate_shape(*7
_class-
+)loc:@generator/batch_normalization_2/beta*
_output_shapes
: 
?
save_1/Assign_78Assign%generator/batch_normalization_2/gammasave_1/RestoreV2:78*
T0*
use_locking(*
validate_shape(*8
_class.
,*loc:@generator/batch_normalization_2/gamma*
_output_shapes
: 
?
save_1/Assign_79Assign*generator/batch_normalization_2/gamma/Adamsave_1/RestoreV2:79*
T0*
use_locking(*
validate_shape(*8
_class.
,*loc:@generator/batch_normalization_2/gamma*
_output_shapes
: 
?
save_1/Assign_80Assign,generator/batch_normalization_2/gamma/Adam_1save_1/RestoreV2:80*
T0*
use_locking(*
validate_shape(*8
_class.
,*loc:@generator/batch_normalization_2/gamma*
_output_shapes
: 
?
save_1/Assign_81Assign+generator/batch_normalization_2/moving_meansave_1/RestoreV2:81*
T0*
use_locking(*
validate_shape(*>
_class4
20loc:@generator/batch_normalization_2/moving_mean*
_output_shapes
: 
?
save_1/Assign_82Assign/generator/batch_normalization_2/moving_variancesave_1/RestoreV2:82*
T0*
use_locking(*
validate_shape(*B
_class8
64loc:@generator/batch_normalization_2/moving_variance*
_output_shapes
: 
?
save_1/Assign_83Assigngenerator/conv2d_transpose/biassave_1/RestoreV2:83*
T0*
use_locking(*
validate_shape(*2
_class(
&$loc:@generator/conv2d_transpose/bias*
_output_shapes
:@
?
save_1/Assign_84Assign$generator/conv2d_transpose/bias/Adamsave_1/RestoreV2:84*
T0*
use_locking(*
validate_shape(*2
_class(
&$loc:@generator/conv2d_transpose/bias*
_output_shapes
:@
?
save_1/Assign_85Assign&generator/conv2d_transpose/bias/Adam_1save_1/RestoreV2:85*
T0*
use_locking(*
validate_shape(*2
_class(
&$loc:@generator/conv2d_transpose/bias*
_output_shapes
:@
?
save_1/Assign_86Assign!generator/conv2d_transpose/kernelsave_1/RestoreV2:86*
T0*
use_locking(*
validate_shape(*4
_class*
(&loc:@generator/conv2d_transpose/kernel*'
_output_shapes
:@?
?
save_1/Assign_87Assign&generator/conv2d_transpose/kernel/Adamsave_1/RestoreV2:87*
T0*
use_locking(*
validate_shape(*4
_class*
(&loc:@generator/conv2d_transpose/kernel*'
_output_shapes
:@?
?
save_1/Assign_88Assign(generator/conv2d_transpose/kernel/Adam_1save_1/RestoreV2:88*
T0*
use_locking(*
validate_shape(*4
_class*
(&loc:@generator/conv2d_transpose/kernel*'
_output_shapes
:@?
?
save_1/Assign_89Assign!generator/conv2d_transpose_1/biassave_1/RestoreV2:89*
T0*
use_locking(*
validate_shape(*4
_class*
(&loc:@generator/conv2d_transpose_1/bias*
_output_shapes
: 
?
save_1/Assign_90Assign&generator/conv2d_transpose_1/bias/Adamsave_1/RestoreV2:90*
T0*
use_locking(*
validate_shape(*4
_class*
(&loc:@generator/conv2d_transpose_1/bias*
_output_shapes
: 
?
save_1/Assign_91Assign(generator/conv2d_transpose_1/bias/Adam_1save_1/RestoreV2:91*
T0*
use_locking(*
validate_shape(*4
_class*
(&loc:@generator/conv2d_transpose_1/bias*
_output_shapes
: 
?
save_1/Assign_92Assign#generator/conv2d_transpose_1/kernelsave_1/RestoreV2:92*
T0*
use_locking(*
validate_shape(*6
_class,
*(loc:@generator/conv2d_transpose_1/kernel*&
_output_shapes
: @
?
save_1/Assign_93Assign(generator/conv2d_transpose_1/kernel/Adamsave_1/RestoreV2:93*
T0*
use_locking(*
validate_shape(*6
_class,
*(loc:@generator/conv2d_transpose_1/kernel*&
_output_shapes
: @
?
save_1/Assign_94Assign*generator/conv2d_transpose_1/kernel/Adam_1save_1/RestoreV2:94*
T0*
use_locking(*
validate_shape(*6
_class,
*(loc:@generator/conv2d_transpose_1/kernel*&
_output_shapes
: @
?
save_1/Assign_95Assign!generator/conv2d_transpose_2/biassave_1/RestoreV2:95*
T0*
use_locking(*
validate_shape(*4
_class*
(&loc:@generator/conv2d_transpose_2/bias*
_output_shapes
:
?
save_1/Assign_96Assign&generator/conv2d_transpose_2/bias/Adamsave_1/RestoreV2:96*
T0*
use_locking(*
validate_shape(*4
_class*
(&loc:@generator/conv2d_transpose_2/bias*
_output_shapes
:
?
save_1/Assign_97Assign(generator/conv2d_transpose_2/bias/Adam_1save_1/RestoreV2:97*
T0*
use_locking(*
validate_shape(*4
_class*
(&loc:@generator/conv2d_transpose_2/bias*
_output_shapes
:
?
save_1/Assign_98Assign#generator/conv2d_transpose_2/kernelsave_1/RestoreV2:98*
T0*
use_locking(*
validate_shape(*6
_class,
*(loc:@generator/conv2d_transpose_2/kernel*&
_output_shapes
: 
?
save_1/Assign_99Assign(generator/conv2d_transpose_2/kernel/Adamsave_1/RestoreV2:99*
T0*
use_locking(*
validate_shape(*6
_class,
*(loc:@generator/conv2d_transpose_2/kernel*&
_output_shapes
: 
?
save_1/Assign_100Assign*generator/conv2d_transpose_2/kernel/Adam_1save_1/RestoreV2:100*
T0*
use_locking(*
validate_shape(*6
_class,
*(loc:@generator/conv2d_transpose_2/kernel*&
_output_shapes
: 
?
save_1/Assign_101Assigngenerator/dense/biassave_1/RestoreV2:101*
T0*
use_locking(*
validate_shape(*'
_class
loc:@generator/dense/bias*
_output_shapes	
:?
?
save_1/Assign_102Assigngenerator/dense/bias/Adamsave_1/RestoreV2:102*
T0*
use_locking(*
validate_shape(*'
_class
loc:@generator/dense/bias*
_output_shapes	
:?
?
save_1/Assign_103Assigngenerator/dense/bias/Adam_1save_1/RestoreV2:103*
T0*
use_locking(*
validate_shape(*'
_class
loc:@generator/dense/bias*
_output_shapes	
:?
?
save_1/Assign_104Assigngenerator/dense/kernelsave_1/RestoreV2:104*
T0*
use_locking(*
validate_shape(*)
_class
loc:@generator/dense/kernel*
_output_shapes
:	d?
?
save_1/Assign_105Assigngenerator/dense/kernel/Adamsave_1/RestoreV2:105*
T0*
use_locking(*
validate_shape(*)
_class
loc:@generator/dense/kernel*
_output_shapes
:	d?
?
save_1/Assign_106Assigngenerator/dense/kernel/Adam_1save_1/RestoreV2:106*
T0*
use_locking(*
validate_shape(*)
_class
loc:@generator/dense/kernel*
_output_shapes
:	d?
?
save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_10^save_1/Assign_100^save_1/Assign_101^save_1/Assign_102^save_1/Assign_103^save_1/Assign_104^save_1/Assign_105^save_1/Assign_106^save_1/Assign_11^save_1/Assign_12^save_1/Assign_13^save_1/Assign_14^save_1/Assign_15^save_1/Assign_16^save_1/Assign_17^save_1/Assign_18^save_1/Assign_19^save_1/Assign_2^save_1/Assign_20^save_1/Assign_21^save_1/Assign_22^save_1/Assign_23^save_1/Assign_24^save_1/Assign_25^save_1/Assign_26^save_1/Assign_27^save_1/Assign_28^save_1/Assign_29^save_1/Assign_3^save_1/Assign_30^save_1/Assign_31^save_1/Assign_32^save_1/Assign_33^save_1/Assign_34^save_1/Assign_35^save_1/Assign_36^save_1/Assign_37^save_1/Assign_38^save_1/Assign_39^save_1/Assign_4^save_1/Assign_40^save_1/Assign_41^save_1/Assign_42^save_1/Assign_43^save_1/Assign_44^save_1/Assign_45^save_1/Assign_46^save_1/Assign_47^save_1/Assign_48^save_1/Assign_49^save_1/Assign_5^save_1/Assign_50^save_1/Assign_51^save_1/Assign_52^save_1/Assign_53^save_1/Assign_54^save_1/Assign_55^save_1/Assign_56^save_1/Assign_57^save_1/Assign_58^save_1/Assign_59^save_1/Assign_6^save_1/Assign_60^save_1/Assign_61^save_1/Assign_62^save_1/Assign_63^save_1/Assign_64^save_1/Assign_65^save_1/Assign_66^save_1/Assign_67^save_1/Assign_68^save_1/Assign_69^save_1/Assign_7^save_1/Assign_70^save_1/Assign_71^save_1/Assign_72^save_1/Assign_73^save_1/Assign_74^save_1/Assign_75^save_1/Assign_76^save_1/Assign_77^save_1/Assign_78^save_1/Assign_79^save_1/Assign_8^save_1/Assign_80^save_1/Assign_81^save_1/Assign_82^save_1/Assign_83^save_1/Assign_84^save_1/Assign_85^save_1/Assign_86^save_1/Assign_87^save_1/Assign_88^save_1/Assign_89^save_1/Assign_9^save_1/Assign_90^save_1/Assign_91^save_1/Assign_92^save_1/Assign_93^save_1/Assign_94^save_1/Assign_95^save_1/Assign_96^save_1/Assign_97^save_1/Assign_98^save_1/Assign_99
1
save_1/restore_allNoOp^save_1/restore_shard"B
save_1/Const:0save_1/Identity:0save_1/restore_all (5 @F8"??
	variables????
H

Variable:0Variable/AssignVariable/read:02Variable/initial_value:0
?
generator/dense/kernel:0generator/dense/kernel/Assigngenerator/dense/kernel/read:023generator/dense/kernel/Initializer/random_uniform:08
~
generator/dense/bias:0generator/dense/bias/Assigngenerator/dense/bias/read:02(generator/dense/bias/Initializer/zeros:08
?
%generator/batch_normalization/gamma:0*generator/batch_normalization/gamma/Assign*generator/batch_normalization/gamma/read:026generator/batch_normalization/gamma/Initializer/ones:08
?
$generator/batch_normalization/beta:0)generator/batch_normalization/beta/Assign)generator/batch_normalization/beta/read:026generator/batch_normalization/beta/Initializer/zeros:08
?
+generator/batch_normalization/moving_mean:00generator/batch_normalization/moving_mean/Assign0generator/batch_normalization/moving_mean/read:02=generator/batch_normalization/moving_mean/Initializer/zeros:0
?
/generator/batch_normalization/moving_variance:04generator/batch_normalization/moving_variance/Assign4generator/batch_normalization/moving_variance/read:02@generator/batch_normalization/moving_variance/Initializer/ones:0
?
#generator/conv2d_transpose/kernel:0(generator/conv2d_transpose/kernel/Assign(generator/conv2d_transpose/kernel/read:02>generator/conv2d_transpose/kernel/Initializer/random_uniform:08
?
!generator/conv2d_transpose/bias:0&generator/conv2d_transpose/bias/Assign&generator/conv2d_transpose/bias/read:023generator/conv2d_transpose/bias/Initializer/zeros:08
?
'generator/batch_normalization_1/gamma:0,generator/batch_normalization_1/gamma/Assign,generator/batch_normalization_1/gamma/read:028generator/batch_normalization_1/gamma/Initializer/ones:08
?
&generator/batch_normalization_1/beta:0+generator/batch_normalization_1/beta/Assign+generator/batch_normalization_1/beta/read:028generator/batch_normalization_1/beta/Initializer/zeros:08
?
-generator/batch_normalization_1/moving_mean:02generator/batch_normalization_1/moving_mean/Assign2generator/batch_normalization_1/moving_mean/read:02?generator/batch_normalization_1/moving_mean/Initializer/zeros:0
?
1generator/batch_normalization_1/moving_variance:06generator/batch_normalization_1/moving_variance/Assign6generator/batch_normalization_1/moving_variance/read:02Bgenerator/batch_normalization_1/moving_variance/Initializer/ones:0
?
%generator/conv2d_transpose_1/kernel:0*generator/conv2d_transpose_1/kernel/Assign*generator/conv2d_transpose_1/kernel/read:02@generator/conv2d_transpose_1/kernel/Initializer/random_uniform:08
?
#generator/conv2d_transpose_1/bias:0(generator/conv2d_transpose_1/bias/Assign(generator/conv2d_transpose_1/bias/read:025generator/conv2d_transpose_1/bias/Initializer/zeros:08
?
'generator/batch_normalization_2/gamma:0,generator/batch_normalization_2/gamma/Assign,generator/batch_normalization_2/gamma/read:028generator/batch_normalization_2/gamma/Initializer/ones:08
?
&generator/batch_normalization_2/beta:0+generator/batch_normalization_2/beta/Assign+generator/batch_normalization_2/beta/read:028generator/batch_normalization_2/beta/Initializer/zeros:08
?
-generator/batch_normalization_2/moving_mean:02generator/batch_normalization_2/moving_mean/Assign2generator/batch_normalization_2/moving_mean/read:02?generator/batch_normalization_2/moving_mean/Initializer/zeros:0
?
1generator/batch_normalization_2/moving_variance:06generator/batch_normalization_2/moving_variance/Assign6generator/batch_normalization_2/moving_variance/read:02Bgenerator/batch_normalization_2/moving_variance/Initializer/ones:0
?
%generator/conv2d_transpose_2/kernel:0*generator/conv2d_transpose_2/kernel/Assign*generator/conv2d_transpose_2/kernel/read:02@generator/conv2d_transpose_2/kernel/Initializer/random_uniform:08
?
#generator/conv2d_transpose_2/bias:0(generator/conv2d_transpose_2/bias/Assign(generator/conv2d_transpose_2/bias/read:025generator/conv2d_transpose_2/bias/Initializer/zeros:08
?
discriminator/conv2d/kernel:0"discriminator/conv2d/kernel/Assign"discriminator/conv2d/kernel/read:028discriminator/conv2d/kernel/Initializer/random_uniform:08
?
discriminator/conv2d/bias:0 discriminator/conv2d/bias/Assign discriminator/conv2d/bias/read:02-discriminator/conv2d/bias/Initializer/zeros:08
?
discriminator/conv2d_1/kernel:0$discriminator/conv2d_1/kernel/Assign$discriminator/conv2d_1/kernel/read:02:discriminator/conv2d_1/kernel/Initializer/random_uniform:08
?
discriminator/conv2d_1/bias:0"discriminator/conv2d_1/bias/Assign"discriminator/conv2d_1/bias/read:02/discriminator/conv2d_1/bias/Initializer/zeros:08
?
discriminator/conv2d_2/kernel:0$discriminator/conv2d_2/kernel/Assign$discriminator/conv2d_2/kernel/read:02:discriminator/conv2d_2/kernel/Initializer/random_uniform:08
?
discriminator/conv2d_2/bias:0"discriminator/conv2d_2/bias/Assign"discriminator/conv2d_2/bias/read:02/discriminator/conv2d_2/bias/Initializer/zeros:08
?
)discriminator/batch_normalization/gamma:0.discriminator/batch_normalization/gamma/Assign.discriminator/batch_normalization/gamma/read:02:discriminator/batch_normalization/gamma/Initializer/ones:0
?
(discriminator/batch_normalization/beta:0-discriminator/batch_normalization/beta/Assign-discriminator/batch_normalization/beta/read:02:discriminator/batch_normalization/beta/Initializer/zeros:0
?
/discriminator/batch_normalization/moving_mean:04discriminator/batch_normalization/moving_mean/Assign4discriminator/batch_normalization/moving_mean/read:02Adiscriminator/batch_normalization/moving_mean/Initializer/zeros:0
?
3discriminator/batch_normalization/moving_variance:08discriminator/batch_normalization/moving_variance/Assign8discriminator/batch_normalization/moving_variance/read:02Ddiscriminator/batch_normalization/moving_variance/Initializer/ones:0
?
discriminator/conv2d_3/kernel:0$discriminator/conv2d_3/kernel/Assign$discriminator/conv2d_3/kernel/read:02:discriminator/conv2d_3/kernel/Initializer/random_uniform:08
?
discriminator/conv2d_3/bias:0"discriminator/conv2d_3/bias/Assign"discriminator/conv2d_3/bias/read:02/discriminator/conv2d_3/bias/Initializer/zeros:08
?
+discriminator/batch_normalization_1/gamma:00discriminator/batch_normalization_1/gamma/Assign0discriminator/batch_normalization_1/gamma/read:02<discriminator/batch_normalization_1/gamma/Initializer/ones:0
?
*discriminator/batch_normalization_1/beta:0/discriminator/batch_normalization_1/beta/Assign/discriminator/batch_normalization_1/beta/read:02<discriminator/batch_normalization_1/beta/Initializer/zeros:0
?
1discriminator/batch_normalization_1/moving_mean:06discriminator/batch_normalization_1/moving_mean/Assign6discriminator/batch_normalization_1/moving_mean/read:02Cdiscriminator/batch_normalization_1/moving_mean/Initializer/zeros:0
?
5discriminator/batch_normalization_1/moving_variance:0:discriminator/batch_normalization_1/moving_variance/Assign:discriminator/batch_normalization_1/moving_variance/read:02Fdiscriminator/batch_normalization_1/moving_variance/Initializer/ones:0
?
discriminator/conv2d_4/kernel:0$discriminator/conv2d_4/kernel/Assign$discriminator/conv2d_4/kernel/read:02:discriminator/conv2d_4/kernel/Initializer/random_uniform:08
?
discriminator/conv2d_4/bias:0"discriminator/conv2d_4/bias/Assign"discriminator/conv2d_4/bias/read:02/discriminator/conv2d_4/bias/Initializer/zeros:08
?
+discriminator/batch_normalization_2/gamma:00discriminator/batch_normalization_2/gamma/Assign0discriminator/batch_normalization_2/gamma/read:02<discriminator/batch_normalization_2/gamma/Initializer/ones:0
?
*discriminator/batch_normalization_2/beta:0/discriminator/batch_normalization_2/beta/Assign/discriminator/batch_normalization_2/beta/read:02<discriminator/batch_normalization_2/beta/Initializer/zeros:0
?
1discriminator/batch_normalization_2/moving_mean:06discriminator/batch_normalization_2/moving_mean/Assign6discriminator/batch_normalization_2/moving_mean/read:02Cdiscriminator/batch_normalization_2/moving_mean/Initializer/zeros:0
?
5discriminator/batch_normalization_2/moving_variance:0:discriminator/batch_normalization_2/moving_variance/Assign:discriminator/batch_normalization_2/moving_variance/read:02Fdiscriminator/batch_normalization_2/moving_variance/Initializer/ones:0
?
discriminator/conv2d_5/kernel:0$discriminator/conv2d_5/kernel/Assign$discriminator/conv2d_5/kernel/read:02:discriminator/conv2d_5/kernel/Initializer/random_uniform:08
?
discriminator/conv2d_5/bias:0"discriminator/conv2d_5/bias/Assign"discriminator/conv2d_5/bias/read:02/discriminator/conv2d_5/bias/Initializer/zeros:08
?
discriminator/dense/kernel:0!discriminator/dense/kernel/Assign!discriminator/dense/kernel/read:027discriminator/dense/kernel/Initializer/random_uniform:08
?
discriminator/dense/bias:0discriminator/dense/bias/Assigndiscriminator/dense/bias/read:02,discriminator/dense/bias/Initializer/zeros:08
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0
?
"discriminator/conv2d/kernel/Adam:0'discriminator/conv2d/kernel/Adam/Assign'discriminator/conv2d/kernel/Adam/read:024discriminator/conv2d/kernel/Adam/Initializer/zeros:0
?
$discriminator/conv2d/kernel/Adam_1:0)discriminator/conv2d/kernel/Adam_1/Assign)discriminator/conv2d/kernel/Adam_1/read:026discriminator/conv2d/kernel/Adam_1/Initializer/zeros:0
?
 discriminator/conv2d/bias/Adam:0%discriminator/conv2d/bias/Adam/Assign%discriminator/conv2d/bias/Adam/read:022discriminator/conv2d/bias/Adam/Initializer/zeros:0
?
"discriminator/conv2d/bias/Adam_1:0'discriminator/conv2d/bias/Adam_1/Assign'discriminator/conv2d/bias/Adam_1/read:024discriminator/conv2d/bias/Adam_1/Initializer/zeros:0
?
$discriminator/conv2d_1/kernel/Adam:0)discriminator/conv2d_1/kernel/Adam/Assign)discriminator/conv2d_1/kernel/Adam/read:026discriminator/conv2d_1/kernel/Adam/Initializer/zeros:0
?
&discriminator/conv2d_1/kernel/Adam_1:0+discriminator/conv2d_1/kernel/Adam_1/Assign+discriminator/conv2d_1/kernel/Adam_1/read:028discriminator/conv2d_1/kernel/Adam_1/Initializer/zeros:0
?
"discriminator/conv2d_1/bias/Adam:0'discriminator/conv2d_1/bias/Adam/Assign'discriminator/conv2d_1/bias/Adam/read:024discriminator/conv2d_1/bias/Adam/Initializer/zeros:0
?
$discriminator/conv2d_1/bias/Adam_1:0)discriminator/conv2d_1/bias/Adam_1/Assign)discriminator/conv2d_1/bias/Adam_1/read:026discriminator/conv2d_1/bias/Adam_1/Initializer/zeros:0
?
$discriminator/conv2d_2/kernel/Adam:0)discriminator/conv2d_2/kernel/Adam/Assign)discriminator/conv2d_2/kernel/Adam/read:026discriminator/conv2d_2/kernel/Adam/Initializer/zeros:0
?
&discriminator/conv2d_2/kernel/Adam_1:0+discriminator/conv2d_2/kernel/Adam_1/Assign+discriminator/conv2d_2/kernel/Adam_1/read:028discriminator/conv2d_2/kernel/Adam_1/Initializer/zeros:0
?
"discriminator/conv2d_2/bias/Adam:0'discriminator/conv2d_2/bias/Adam/Assign'discriminator/conv2d_2/bias/Adam/read:024discriminator/conv2d_2/bias/Adam/Initializer/zeros:0
?
$discriminator/conv2d_2/bias/Adam_1:0)discriminator/conv2d_2/bias/Adam_1/Assign)discriminator/conv2d_2/bias/Adam_1/read:026discriminator/conv2d_2/bias/Adam_1/Initializer/zeros:0
?
$discriminator/conv2d_3/kernel/Adam:0)discriminator/conv2d_3/kernel/Adam/Assign)discriminator/conv2d_3/kernel/Adam/read:026discriminator/conv2d_3/kernel/Adam/Initializer/zeros:0
?
&discriminator/conv2d_3/kernel/Adam_1:0+discriminator/conv2d_3/kernel/Adam_1/Assign+discriminator/conv2d_3/kernel/Adam_1/read:028discriminator/conv2d_3/kernel/Adam_1/Initializer/zeros:0
?
"discriminator/conv2d_3/bias/Adam:0'discriminator/conv2d_3/bias/Adam/Assign'discriminator/conv2d_3/bias/Adam/read:024discriminator/conv2d_3/bias/Adam/Initializer/zeros:0
?
$discriminator/conv2d_3/bias/Adam_1:0)discriminator/conv2d_3/bias/Adam_1/Assign)discriminator/conv2d_3/bias/Adam_1/read:026discriminator/conv2d_3/bias/Adam_1/Initializer/zeros:0
?
$discriminator/conv2d_4/kernel/Adam:0)discriminator/conv2d_4/kernel/Adam/Assign)discriminator/conv2d_4/kernel/Adam/read:026discriminator/conv2d_4/kernel/Adam/Initializer/zeros:0
?
&discriminator/conv2d_4/kernel/Adam_1:0+discriminator/conv2d_4/kernel/Adam_1/Assign+discriminator/conv2d_4/kernel/Adam_1/read:028discriminator/conv2d_4/kernel/Adam_1/Initializer/zeros:0
?
"discriminator/conv2d_4/bias/Adam:0'discriminator/conv2d_4/bias/Adam/Assign'discriminator/conv2d_4/bias/Adam/read:024discriminator/conv2d_4/bias/Adam/Initializer/zeros:0
?
$discriminator/conv2d_4/bias/Adam_1:0)discriminator/conv2d_4/bias/Adam_1/Assign)discriminator/conv2d_4/bias/Adam_1/read:026discriminator/conv2d_4/bias/Adam_1/Initializer/zeros:0
?
$discriminator/conv2d_5/kernel/Adam:0)discriminator/conv2d_5/kernel/Adam/Assign)discriminator/conv2d_5/kernel/Adam/read:026discriminator/conv2d_5/kernel/Adam/Initializer/zeros:0
?
&discriminator/conv2d_5/kernel/Adam_1:0+discriminator/conv2d_5/kernel/Adam_1/Assign+discriminator/conv2d_5/kernel/Adam_1/read:028discriminator/conv2d_5/kernel/Adam_1/Initializer/zeros:0
?
"discriminator/conv2d_5/bias/Adam:0'discriminator/conv2d_5/bias/Adam/Assign'discriminator/conv2d_5/bias/Adam/read:024discriminator/conv2d_5/bias/Adam/Initializer/zeros:0
?
$discriminator/conv2d_5/bias/Adam_1:0)discriminator/conv2d_5/bias/Adam_1/Assign)discriminator/conv2d_5/bias/Adam_1/read:026discriminator/conv2d_5/bias/Adam_1/Initializer/zeros:0
?
!discriminator/dense/kernel/Adam:0&discriminator/dense/kernel/Adam/Assign&discriminator/dense/kernel/Adam/read:023discriminator/dense/kernel/Adam/Initializer/zeros:0
?
#discriminator/dense/kernel/Adam_1:0(discriminator/dense/kernel/Adam_1/Assign(discriminator/dense/kernel/Adam_1/read:025discriminator/dense/kernel/Adam_1/Initializer/zeros:0
?
discriminator/dense/bias/Adam:0$discriminator/dense/bias/Adam/Assign$discriminator/dense/bias/Adam/read:021discriminator/dense/bias/Adam/Initializer/zeros:0
?
!discriminator/dense/bias/Adam_1:0&discriminator/dense/bias/Adam_1/Assign&discriminator/dense/bias/Adam_1/read:023discriminator/dense/bias/Adam_1/Initializer/zeros:0
\
beta1_power_1:0beta1_power_1/Assignbeta1_power_1/read:02beta1_power_1/initial_value:0
\
beta2_power_1:0beta2_power_1/Assignbeta2_power_1/read:02beta2_power_1/initial_value:0
?
generator/dense/kernel/Adam:0"generator/dense/kernel/Adam/Assign"generator/dense/kernel/Adam/read:02/generator/dense/kernel/Adam/Initializer/zeros:0
?
generator/dense/kernel/Adam_1:0$generator/dense/kernel/Adam_1/Assign$generator/dense/kernel/Adam_1/read:021generator/dense/kernel/Adam_1/Initializer/zeros:0
?
generator/dense/bias/Adam:0 generator/dense/bias/Adam/Assign generator/dense/bias/Adam/read:02-generator/dense/bias/Adam/Initializer/zeros:0
?
generator/dense/bias/Adam_1:0"generator/dense/bias/Adam_1/Assign"generator/dense/bias/Adam_1/read:02/generator/dense/bias/Adam_1/Initializer/zeros:0
?
*generator/batch_normalization/gamma/Adam:0/generator/batch_normalization/gamma/Adam/Assign/generator/batch_normalization/gamma/Adam/read:02<generator/batch_normalization/gamma/Adam/Initializer/zeros:0
?
,generator/batch_normalization/gamma/Adam_1:01generator/batch_normalization/gamma/Adam_1/Assign1generator/batch_normalization/gamma/Adam_1/read:02>generator/batch_normalization/gamma/Adam_1/Initializer/zeros:0
?
)generator/batch_normalization/beta/Adam:0.generator/batch_normalization/beta/Adam/Assign.generator/batch_normalization/beta/Adam/read:02;generator/batch_normalization/beta/Adam/Initializer/zeros:0
?
+generator/batch_normalization/beta/Adam_1:00generator/batch_normalization/beta/Adam_1/Assign0generator/batch_normalization/beta/Adam_1/read:02=generator/batch_normalization/beta/Adam_1/Initializer/zeros:0
?
(generator/conv2d_transpose/kernel/Adam:0-generator/conv2d_transpose/kernel/Adam/Assign-generator/conv2d_transpose/kernel/Adam/read:02:generator/conv2d_transpose/kernel/Adam/Initializer/zeros:0
?
*generator/conv2d_transpose/kernel/Adam_1:0/generator/conv2d_transpose/kernel/Adam_1/Assign/generator/conv2d_transpose/kernel/Adam_1/read:02<generator/conv2d_transpose/kernel/Adam_1/Initializer/zeros:0
?
&generator/conv2d_transpose/bias/Adam:0+generator/conv2d_transpose/bias/Adam/Assign+generator/conv2d_transpose/bias/Adam/read:028generator/conv2d_transpose/bias/Adam/Initializer/zeros:0
?
(generator/conv2d_transpose/bias/Adam_1:0-generator/conv2d_transpose/bias/Adam_1/Assign-generator/conv2d_transpose/bias/Adam_1/read:02:generator/conv2d_transpose/bias/Adam_1/Initializer/zeros:0
?
,generator/batch_normalization_1/gamma/Adam:01generator/batch_normalization_1/gamma/Adam/Assign1generator/batch_normalization_1/gamma/Adam/read:02>generator/batch_normalization_1/gamma/Adam/Initializer/zeros:0
?
.generator/batch_normalization_1/gamma/Adam_1:03generator/batch_normalization_1/gamma/Adam_1/Assign3generator/batch_normalization_1/gamma/Adam_1/read:02@generator/batch_normalization_1/gamma/Adam_1/Initializer/zeros:0
?
+generator/batch_normalization_1/beta/Adam:00generator/batch_normalization_1/beta/Adam/Assign0generator/batch_normalization_1/beta/Adam/read:02=generator/batch_normalization_1/beta/Adam/Initializer/zeros:0
?
-generator/batch_normalization_1/beta/Adam_1:02generator/batch_normalization_1/beta/Adam_1/Assign2generator/batch_normalization_1/beta/Adam_1/read:02?generator/batch_normalization_1/beta/Adam_1/Initializer/zeros:0
?
*generator/conv2d_transpose_1/kernel/Adam:0/generator/conv2d_transpose_1/kernel/Adam/Assign/generator/conv2d_transpose_1/kernel/Adam/read:02<generator/conv2d_transpose_1/kernel/Adam/Initializer/zeros:0
?
,generator/conv2d_transpose_1/kernel/Adam_1:01generator/conv2d_transpose_1/kernel/Adam_1/Assign1generator/conv2d_transpose_1/kernel/Adam_1/read:02>generator/conv2d_transpose_1/kernel/Adam_1/Initializer/zeros:0
?
(generator/conv2d_transpose_1/bias/Adam:0-generator/conv2d_transpose_1/bias/Adam/Assign-generator/conv2d_transpose_1/bias/Adam/read:02:generator/conv2d_transpose_1/bias/Adam/Initializer/zeros:0
?
*generator/conv2d_transpose_1/bias/Adam_1:0/generator/conv2d_transpose_1/bias/Adam_1/Assign/generator/conv2d_transpose_1/bias/Adam_1/read:02<generator/conv2d_transpose_1/bias/Adam_1/Initializer/zeros:0
?
,generator/batch_normalization_2/gamma/Adam:01generator/batch_normalization_2/gamma/Adam/Assign1generator/batch_normalization_2/gamma/Adam/read:02>generator/batch_normalization_2/gamma/Adam/Initializer/zeros:0
?
.generator/batch_normalization_2/gamma/Adam_1:03generator/batch_normalization_2/gamma/Adam_1/Assign3generator/batch_normalization_2/gamma/Adam_1/read:02@generator/batch_normalization_2/gamma/Adam_1/Initializer/zeros:0
?
+generator/batch_normalization_2/beta/Adam:00generator/batch_normalization_2/beta/Adam/Assign0generator/batch_normalization_2/beta/Adam/read:02=generator/batch_normalization_2/beta/Adam/Initializer/zeros:0
?
-generator/batch_normalization_2/beta/Adam_1:02generator/batch_normalization_2/beta/Adam_1/Assign2generator/batch_normalization_2/beta/Adam_1/read:02?generator/batch_normalization_2/beta/Adam_1/Initializer/zeros:0
?
*generator/conv2d_transpose_2/kernel/Adam:0/generator/conv2d_transpose_2/kernel/Adam/Assign/generator/conv2d_transpose_2/kernel/Adam/read:02<generator/conv2d_transpose_2/kernel/Adam/Initializer/zeros:0
?
,generator/conv2d_transpose_2/kernel/Adam_1:01generator/conv2d_transpose_2/kernel/Adam_1/Assign1generator/conv2d_transpose_2/kernel/Adam_1/read:02>generator/conv2d_transpose_2/kernel/Adam_1/Initializer/zeros:0
?
(generator/conv2d_transpose_2/bias/Adam:0-generator/conv2d_transpose_2/bias/Adam/Assign-generator/conv2d_transpose_2/bias/Adam/read:02:generator/conv2d_transpose_2/bias/Adam/Initializer/zeros:0
?
*generator/conv2d_transpose_2/bias/Adam_1:0/generator/conv2d_transpose_2/bias/Adam_1/Assign/generator/conv2d_transpose_2/bias/Adam_1/read:02<generator/conv2d_transpose_2/bias/Adam_1/Initializer/zeros:0"?%
trainable_variables?%?%
?
generator/dense/kernel:0generator/dense/kernel/Assigngenerator/dense/kernel/read:023generator/dense/kernel/Initializer/random_uniform:08
~
generator/dense/bias:0generator/dense/bias/Assigngenerator/dense/bias/read:02(generator/dense/bias/Initializer/zeros:08
?
%generator/batch_normalization/gamma:0*generator/batch_normalization/gamma/Assign*generator/batch_normalization/gamma/read:026generator/batch_normalization/gamma/Initializer/ones:08
?
$generator/batch_normalization/beta:0)generator/batch_normalization/beta/Assign)generator/batch_normalization/beta/read:026generator/batch_normalization/beta/Initializer/zeros:08
?
#generator/conv2d_transpose/kernel:0(generator/conv2d_transpose/kernel/Assign(generator/conv2d_transpose/kernel/read:02>generator/conv2d_transpose/kernel/Initializer/random_uniform:08
?
!generator/conv2d_transpose/bias:0&generator/conv2d_transpose/bias/Assign&generator/conv2d_transpose/bias/read:023generator/conv2d_transpose/bias/Initializer/zeros:08
?
'generator/batch_normalization_1/gamma:0,generator/batch_normalization_1/gamma/Assign,generator/batch_normalization_1/gamma/read:028generator/batch_normalization_1/gamma/Initializer/ones:08
?
&generator/batch_normalization_1/beta:0+generator/batch_normalization_1/beta/Assign+generator/batch_normalization_1/beta/read:028generator/batch_normalization_1/beta/Initializer/zeros:08
?
%generator/conv2d_transpose_1/kernel:0*generator/conv2d_transpose_1/kernel/Assign*generator/conv2d_transpose_1/kernel/read:02@generator/conv2d_transpose_1/kernel/Initializer/random_uniform:08
?
#generator/conv2d_transpose_1/bias:0(generator/conv2d_transpose_1/bias/Assign(generator/conv2d_transpose_1/bias/read:025generator/conv2d_transpose_1/bias/Initializer/zeros:08
?
'generator/batch_normalization_2/gamma:0,generator/batch_normalization_2/gamma/Assign,generator/batch_normalization_2/gamma/read:028generator/batch_normalization_2/gamma/Initializer/ones:08
?
&generator/batch_normalization_2/beta:0+generator/batch_normalization_2/beta/Assign+generator/batch_normalization_2/beta/read:028generator/batch_normalization_2/beta/Initializer/zeros:08
?
%generator/conv2d_transpose_2/kernel:0*generator/conv2d_transpose_2/kernel/Assign*generator/conv2d_transpose_2/kernel/read:02@generator/conv2d_transpose_2/kernel/Initializer/random_uniform:08
?
#generator/conv2d_transpose_2/bias:0(generator/conv2d_transpose_2/bias/Assign(generator/conv2d_transpose_2/bias/read:025generator/conv2d_transpose_2/bias/Initializer/zeros:08
?
discriminator/conv2d/kernel:0"discriminator/conv2d/kernel/Assign"discriminator/conv2d/kernel/read:028discriminator/conv2d/kernel/Initializer/random_uniform:08
?
discriminator/conv2d/bias:0 discriminator/conv2d/bias/Assign discriminator/conv2d/bias/read:02-discriminator/conv2d/bias/Initializer/zeros:08
?
discriminator/conv2d_1/kernel:0$discriminator/conv2d_1/kernel/Assign$discriminator/conv2d_1/kernel/read:02:discriminator/conv2d_1/kernel/Initializer/random_uniform:08
?
discriminator/conv2d_1/bias:0"discriminator/conv2d_1/bias/Assign"discriminator/conv2d_1/bias/read:02/discriminator/conv2d_1/bias/Initializer/zeros:08
?
discriminator/conv2d_2/kernel:0$discriminator/conv2d_2/kernel/Assign$discriminator/conv2d_2/kernel/read:02:discriminator/conv2d_2/kernel/Initializer/random_uniform:08
?
discriminator/conv2d_2/bias:0"discriminator/conv2d_2/bias/Assign"discriminator/conv2d_2/bias/read:02/discriminator/conv2d_2/bias/Initializer/zeros:08
?
discriminator/conv2d_3/kernel:0$discriminator/conv2d_3/kernel/Assign$discriminator/conv2d_3/kernel/read:02:discriminator/conv2d_3/kernel/Initializer/random_uniform:08
?
discriminator/conv2d_3/bias:0"discriminator/conv2d_3/bias/Assign"discriminator/conv2d_3/bias/read:02/discriminator/conv2d_3/bias/Initializer/zeros:08
?
discriminator/conv2d_4/kernel:0$discriminator/conv2d_4/kernel/Assign$discriminator/conv2d_4/kernel/read:02:discriminator/conv2d_4/kernel/Initializer/random_uniform:08
?
discriminator/conv2d_4/bias:0"discriminator/conv2d_4/bias/Assign"discriminator/conv2d_4/bias/read:02/discriminator/conv2d_4/bias/Initializer/zeros:08
?
discriminator/conv2d_5/kernel:0$discriminator/conv2d_5/kernel/Assign$discriminator/conv2d_5/kernel/read:02:discriminator/conv2d_5/kernel/Initializer/random_uniform:08
?
discriminator/conv2d_5/bias:0"discriminator/conv2d_5/bias/Assign"discriminator/conv2d_5/bias/read:02/discriminator/conv2d_5/bias/Initializer/zeros:08
?
discriminator/dense/kernel:0!discriminator/dense/kernel/Assign!discriminator/dense/kernel/read:027discriminator/dense/kernel/Initializer/random_uniform:08
?
discriminator/dense/bias:0discriminator/dense/bias/Assigndiscriminator/dense/bias/read:02,discriminator/dense/bias/Initializer/zeros:08"?

update_ops?
?
-generator/batch_normalization/AssignMovingAvg
/generator/batch_normalization/AssignMovingAvg_1
/generator/batch_normalization_1/AssignMovingAvg
1generator/batch_normalization_1/AssignMovingAvg_1
/generator/batch_normalization_2/AssignMovingAvg
1generator/batch_normalization_2/AssignMovingAvg_1"
train_op

Adam
Adam_1*?
serving_default?
6
x_input+
input_real:0?????????  6
y_output*
discriminator/out:0??????????tensorflow/serving/predict