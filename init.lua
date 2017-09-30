require('nn')

nnst = {} -- define the global nnst table

include('Linear.lua')

include('TransferFunction.lua')
include('TransferFunctionALU.lua')
include('TransferFunctionReCU.lua')
include('TransferFunctionFactory.lua')

include('NetworkLayer.lua')
include('NetworkSequential.lua')
include('NetworkScalarFunction.lua')
include('NetworkImporter.lua')

include('ErrorCriterion.lua')
include('ErrorCriterionSquare.lua')
include('ErrorCriterionSquareRelative.lua')
include('ErrorCriterionRelative.lua')

include('ApproximationNormalizatorFactory.lua')

include('TrainerForScalarSequence.lua')

include('ExportPercepton.lua')
include('TrainerExportToCsv.lua')

return nnst