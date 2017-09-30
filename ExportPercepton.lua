local ExportPercepton = torch.class('nnst.ExportPercepton')

function ExportPercepton:__init()
  self.states = {}

  table.insert(self.states, {
    X = "X",
    A = "A",
    B = "B",
    Ys = "Ys",
    Ya = "Ya",
    Ye = "Ye",
    E = "E",
    dA = "dA",
    dB = "dB",
    ddA = "ddA",
    ddB = "ddB",    
    uA = "uA",
    uB = "uB",
    de_Ya = "de/dYa",
    dde_Ya = "dde/ddYa",
    de_Ys = "dYe/dYs",    
    dde_Ys = "ddYe/ddYs",
    dYa_Ys = "dYa/dYs",    
    ddYa_Ys = "ddYa/ddYs"
  })
end

function ExportPercepton:addState(xInput, yExpected, error, de, dde, net)
  local layer = net:getInputLayer()

  local res = {
    X = xInput,
    A = layer:getWeight(1, 1),
    B = layer:getBias(1),
    Ys = layer:getLinearOutput(1),
    Ya = layer:getTransferOutput(1),
    Ye = yExpected,
    E = error,
    dA = layer:getWeightGrad(1, 1),
    dB = layer:getBiasGrad(1),
    ddA = layer:getWeightGrad2(1, 1),
    ddB = layer:getBiasGrad2(1),
    de_Ya = de,    
    dde_Ya = dde,
    de_Ys = layer:getTransferGradInput(1),
    dde_Ys = layer:getTransferGrad2Input(1),
    dYa_Ys = layer:getTransferGradForInput(1),
    ddYa_Ys = layer:getTransferGrad2ForInput(1)
  }

  res.uA = res.dA / res.ddA
  res.uB = res.dB / res.ddB

  table.insert(self.states, res)
end

function ExportPercepton:exportToCsv(path)
  fd = io.open(path, 'w')

  for i=1,table.getn(self.states) do
      local state = self.states[i]      
      fd:write(state.X, ",", state.A, ",", state.B, ",", state.Ys, ",", state.Ya, ",", state.Ye, ",", state.E)
      fd:write(",", state.uA, ",", state.uB)
      fd:write(",", state.de_Ya, ", ", state.de_Ys, ",", state.dYa_Ys, ",", state.dA, ",", state.dB)
      fd:write(",", state.dde_Ya, ", ", state.dde_Ys, ",", state.ddYa_Ys, ",", state.ddA, ",", state.ddB)
      fd:write("\n")
  end

  fd:close()
end