clear; clc;

base_dir = '/home/yijui/Downloads/cost2100-master';

raw_files = {
    'D1_raw.mat', 'data_D1';
    'D2_raw.mat', 'data_D2';
    'D3_raw.mat', 'data_D3';
    'D4_raw.mat', 'data_D4';
    'D5_raw.mat', 'data_D5';
    'D6_raw.mat', 'data_D6';
};

for d = 1:size(raw_files, 1)

    raw_path = fullfile(base_dir, raw_files{d, 1});
    out_dir  = fullfile(base_dir, raw_files{d, 2});

    fprintf('\nConverting %s -> %s\n', raw_files{d, 1}, raw_files{d, 2});

    load(raw_path, 'H_transfer');

    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    [numSnap, numFreq, numUsers, numAnt] = size(H_transfer);

    Nc = 32;
    Nt = 32;
    C = 2;

    if numFreq < 125
        error('%s has fewer than 125 frequency bins.', raw_files{d, 1});
    end

    if numAnt < 32
        error('%s has fewer than 32 antenna links.', raw_files{d, 1});
    end

    N = numSnap * numUsers;
    HT_all = zeros(N, C * Nc * Nt, 'single');

    idx = 1;

    for u = 1:numUsers
        for s = 1:numSnap

            H = squeeze(H_transfer(s, 1:Nc, u, 1:Nt));  % 32 x 32 complex

            scale = max(abs(H(:))) + 1e-12;
            H = H / (2 * scale);

            H_real = real(H) + 0.5;
            H_imag = imag(H) + 0.5;

            H_stack = cat(3, H_real, H_imag);
            H_stack = permute(H_stack, [3, 1, 2]);

            HT_all(idx, :) = reshape(H_stack, 1, []);

            idx = idx + 1;
        end
    end

    rng(42);
    perm = randperm(N);

    nTrain = floor(0.8 * N);
    nVal = floor(0.1 * N);

    idxTrain = perm(1:nTrain);
    idxVal = perm(nTrain+1:nTrain+nVal);
    idxTest = perm(nTrain+nVal+1:end);

    HT = HT_all(idxTrain, :);
    save(fullfile(out_dir, 'DATA_Htrainin.mat'), 'HT', '-v7.3');

    HT = HT_all(idxVal, :);
    save(fullfile(out_dir, 'DATA_Hvalin.mat'), 'HT', '-v7.3');

    HT = HT_all(idxTest, :);
    save(fullfile(out_dir, 'DATA_Htestin.mat'), 'HT', '-v7.3');

    testN = length(idxTest);
    HF_all = zeros(testN, 32, 125);

    for k = 1:testN

        linearIdx = idxTest(k);

        u = ceil(linearIdx / numSnap);
        s = linearIdx - (u - 1) * numSnap;

        Hf = squeeze(H_transfer(s, 1:125, u, 1:32));  % 125 x 32
        HF_all(k, :, :) = Hf.';                       % 32 x 125
    end

    save(fullfile(out_dir, 'DATA_HtestFin_all.mat'), 'HF_all', '-v7.3');

    fprintf('Saved CsiNet data to: %s\n', out_dir);
    fprintf('Train = %d, Val = %d, Test = %d\n', ...
        length(idxTrain), length(idxVal), length(idxTest));

    clear H_transfer HT_all HT HF_all
end

fprintf('\nAll D1 ~ D6 converted to CsiNet format.\n');