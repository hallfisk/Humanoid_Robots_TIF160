function fitness = EvaluateIndividual(chromosome, functionData, constantRegister, nrOfVariableRegisters)
       
    nrOfDataPoints = length(functionData);
    
    error = 0;
    for k = 1:nrOfDataPoints
        x = functionData(k,1);
        y = functionData(k,2);
        yEstimate = CalculateEstimation(chromosome, x, nrOfVariableRegisters, constantRegister);
        error = error + (yEstimate - y)^2;
    end

    error = (error / nrOfDataPoints)^0.5;

    if isnan(error)
        error = intmax;
    end

    fitness = 1 / error;
    
end