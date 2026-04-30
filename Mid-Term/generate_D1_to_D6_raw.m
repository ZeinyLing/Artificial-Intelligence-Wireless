% =========================================================
% 2. Define D1 ~ D6 settings
% =========================================================
datasets = struct([]);

Nms = 9;

% ---------------------------------------------------------
% D1: Indoor, LOS, original closely-spaced users
% ---------------------------------------------------------
datasets(1).name = 'D1';
datasets(1).Network = 'Indoor_CloselySpacedUser_2_6GHz';
datasets(1).scenario = 'LOS';
datasets(1).freq = [2.58e9 2.62e9];
datasets(1).snapNum = 50;
datasets(1).snapRate = 50;
datasets(1).BSPosCenter = [0.30 -4.37 3.20];
datasets(1).BSPosSpacing = [0 0 0];
datasets(1).BSPosNum = 1;

% 原始 indoor closely-spaced users，可用 demo_model 的 closely-spaced setting
datasets(1).MSPos = [-1.5, -1.5, 2.5; ...
                      0.0, -1.5, 2.5; ...
                      1.5, -1.5, 2.5; ...
                     -1.5,  0.0, 2.5; ...
                      0.0,  0.0, 2.5; ...
                      1.5,  0.0, 2.5; ...
                     -1.5,  1.5, 2.5; ...
                      0.0,  1.5, 2.5; ...
                      1.5,  1.5, 2.5];

datasets(1).MSVelo = repmat([-.25, 0, 0], Nms, 1);

% ---------------------------------------------------------
% D2: Indoor, LOS, center-clustered users
% ---------------------------------------------------------
datasets(2).name = 'D2';
datasets(2).Network = 'Indoor_CloselySpacedUser_2_6GHz';
datasets(2).scenario = 'LOS';
datasets(2).freq = [2.58e9 2.62e9];
datasets(2).snapNum = 50;
datasets(2).snapRate = 50;
datasets(2).BSPosCenter = [0.30 -4.37 3.20];
datasets(2).BSPosSpacing = [0 0 0];
datasets(2).BSPosNum = 1;

MSPos = zeros(Nms, 3);
MSPos(:,1) = randn(Nms,1) * 0.8;
MSPos(:,2) = randn(Nms,1) * 0.8;
MSPos(:,3) = 2.5 + randn(Nms,1) * 0.2;
datasets(2).MSPos = MSPos;
datasets(2).MSVelo = repmat([-.25, 0, 0], Nms, 1);

% ---------------------------------------------------------
% D3: Indoor, LOS, spread / edge users
% ---------------------------------------------------------
datasets(3).name = 'D3';
datasets(3).Network = 'Indoor_CloselySpacedUser_2_6GHz';
datasets(3).scenario = 'LOS';
datasets(3).freq = [2.58e9 2.62e9];
datasets(3).snapNum = 50;
datasets(3).snapRate = 50;
datasets(3).BSPosCenter = [0.30 -4.37 3.20];
datasets(3).BSPosSpacing = [0 0 0];
datasets(3).BSPosNum = 1;

theta = rand(Nms,1) * 2 * pi;
r = 3 + rand(Nms,1) * 2;
MSPos = zeros(Nms, 3);
MSPos(:,1) = r .* cos(theta);
MSPos(:,2) = r .* sin(theta);
MSPos(:,3) = 2.5 + randn(Nms,1) * 0.2;
datasets(3).MSPos = MSPos;
datasets(3).MSVelo = repmat([-.25, 0, 0], Nms, 1);

% ---------------------------------------------------------
% D4: SemiUrban, LOS, original closely-spaced users
% ---------------------------------------------------------
datasets(4).name = 'D4';
datasets(4).Network = 'SemiUrban_CloselySpacedUser_2_6GHz';
datasets(4).scenario = 'LOS';
datasets(4).freq = [2.58e9 2.62e9];
datasets(4).snapRate = 50;
datasets(4).snapNum = 50;
datasets(4).MSPos = [-27, -10, 0; ...
                     -27-1.5, -10+1.5, 0; ...
                     -27, -10+1.5, 0; ...
                     -27+1.5, -10+1.5, 0; ...
                     -27-1.5, -10, 0; ...
                     -27+1.5, -10, 0; ...
                     -27-1.5, -10-1.5, 0; ...
                     -27, -10-1.5, 0; ...
                     -27+1.5, -10-1.5, 0];

datasets(4).MSVelo = [0.4, 0.3, 0; ...
                      0.3, -0.4, 0; ...
                      -0.5, 0.1, 0; ...
                      -0.3, -0.4, 0; ...
                      -0.4, -0.2, 0; ...
                      0.3, 0.4, 0; ...
                      0.3, -0.3, 0; ...
                      -0.45, 0.1, 0; ...
                      0.4, -0.3, 0];

datasets(4).BSPosCenter = [0 0 8];
datasets(4).BSPosSpacing = [0 0 0];
datasets(4).BSPosNum = 1;

% ---------------------------------------------------------
% D5: SemiUrban, LOS, well-separated users
% ---------------------------------------------------------
datasets(5).name = 'D5';
datasets(5).Network = 'SemiUrban_CloselySpacedUser_2_6GHz';
datasets(5).scenario = 'LOS';
datasets(5).freq = [2.58e9 2.62e9];
datasets(5).snapRate = 50;
datasets(5).snapNum = 50;

datasets(5).MSPos = [27, 10, 0; ...
                     -27, 10, 0; ...
                     -27, -10, 0; ...
                     27, -10, 0; ...
                     10, 27, 0; ...
                     -10, 27, 0; ...
                     -10, -27, 0; ...
                     10, -27, 0; ...
                     5, 20, 0];

datasets(5).MSVelo = datasets(4).MSVelo;
datasets(5).BSPosCenter = [0 0 8];
datasets(5).BSPosSpacing = [0 0 0];
datasets(5).BSPosNum = 1;

% ---------------------------------------------------------
% D6: SemiUrban, NLOS, well-separated users
% ---------------------------------------------------------
datasets(6) = datasets(5);
datasets(6).name = 'D6';
datasets(6).scenario = 'NLOS';

% =========================================================
% 3. Generate D1 ~ D6
% =========================================================