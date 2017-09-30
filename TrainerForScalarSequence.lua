local inspect = require('inspect');
local TrainerForScalarSequence, parent = torch.class('nnst.TrainerForScalarSequence')

local function addToLearningHistory(trainer, input, valueExpected, valuePredicted, learningCriterion)
    local state = {
        input = input,
        valueExpected = valueExpected,
        valuePredicted = valuePredicted,
        errorCriterion = learningCriterion:forward(torch.Tensor{valuePredicted}, torch.Tensor{valueExpected})
    }

    table.insert(trainer.learningHistory, state)
    return state
end

local function supervisedLearning(trainer, learningRate, learningSession)
    local inputTensor = torch.Tensor{ learningSession.input }
    local predictedOutputTensor = torch.Tensor{ learningSession.valuePredicted }
    local expectedOutputTensor = torch.Tensor{ learningSession.valueExpected }

    local gradErrorCriterion = trainer.config.learningCriterion:backward(predictedOutputTensor, expectedOutputTensor)

    trainer.net:zeroGradParameters()
    trainer.net:backward(inputTensor, gradErrorCriterion)

    learningSession.gradErrorCriterion = torch.totable(gradErrorCriterion)
    learningSession.net = trainer.net:getState()
    learningSession.learningRate = learningRate

    trainer.net:updateParameters(learningRate)
end

local function addToExperimentsHistory(trainer, input, valueExpected)
    local point = {
        input = input,
        value = valueExpected
    }

    table.insert(trainer.experimentsHistory, point)
    return point
end

local function learnFromExperiment(trainer, learningRate, point)
    local valuePredicted = trainer.net:forwardScalar(point.input)
    local session = addToLearningHistory(trainer, point.input, point.value, valuePredicted, trainer.config.learningCriterion)
    supervisedLearning(trainer, learningRate, session)
end

--
-- TrainerForScalarSequence
--

function TrainerForScalarSequence:__init(args)
    self.net = args.net

    self.config = {
        learningCriterion = args.criterion or nnst.RelativeErrorCriterion(),
        learningRate = args.learningRate or 0.1,        

        stopConditionMaxError = args.stopConditionMaxError or 0.1,
        stopConditionGoodPreditionsCount = args.stopConditionGoodPreditionsCount or 5
    }

    self.experimentsHistory = {}

    self.learningHistory = {}
end

function TrainerForScalarSequence:getLearningSessionsCount()
    return table.getn(self.learningHistory)
end

function TrainerForScalarSequence:getLastLearningSessionsErrors(aCount)
    local resultsCount = aCount or self.config.stopConditionPreditionsCount
    local learningsCount = self:getLearningSessionsCount();
    if resultsCount > learningsCount then
        resultsCount = learningsCount
    end

    local result = torch.Tensor(resultsCount)
    for i=1,resultsCount do
        result[i] = self.learningHistory[learningsCount - i + 1].errorCriterion
    end

    return result
end

function TrainerForScalarSequence:hasLearnedSequence()
    if self:getLearningSessionsCount() < self.config.stopConditionGoodPreditionsCount then
        return false
    end

    local maxError = torch.max(torch.abs(self:getLastLearningSessionsErrors(self.config.stopConditionGoodPreditionsCount)))
    return maxError <= self.config.stopConditionMaxError
end

function TrainerForScalarSequence:getExperimentsCount()
    return table.getn(self.experimentsHistory)
end

function TrainerForScalarSequence:learnFromSequence(seq)
    local inputValue = seq:getCurrentValue()
    local expectedOutputValue = seq:getNextValue()
    addToExperimentsHistory(self, inputValue, expectedOutputValue)

    local predictedOutputValue = self.net:forwardScalar(inputValue)
    local learningSession = addToLearningHistory(self, inputValue, expectedOutputValue, predictedOutputValue, self.config.learningCriterion)

    if not self:hasLearnedSequence() then
        supervisedLearning(self, self.config.learningRate, learningSession)        
    end

    return self:hasLearnedSequence()
end

function TrainerForScalarSequence:repeatLastLesson()
    local experimentsCount = self:getExperimentsCount();
    assert(experimentsCount > 0);

    local experiment = self.experimentsHistory[experimentsCount]
    learnFromExperiment(self, self.config.learningRate, experiment);

    return self:hasLearnedSequence()
end

function TrainerForScalarSequence:getLearningHistory()
    return self.learningHistory
end

function TrainerForScalarSequence:getLearningHistoryDescription()
    return inspect(self.learningHistory)
end