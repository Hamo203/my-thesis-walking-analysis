% 読み込み
T = readtable("summary_knee_vGRF_QC_kai_ver4.csv"); % File,KneeMax,KneeMin,ROM_total,ROM_swing,vGRFmax_norm,...
T_use = T(T.use_this == 1, :);

% 膝角度peek,ROM,vgrfの分布と分位数
figure;
subplot(2,1,1);
histogram(T_use.KneeMax(~isnan(T_use.KneeMax)),50);
xlabel('Peak knee flexion (deg)'); title('Peak knee flexion during swing (use this = 1)');

subplot(2,1,2);
histogram(T_use.ROM_swing(~isnan(T_use.ROM_swing)),50);
xlabel('ROM during swing (deg)'); title('Swing ROM (use this = 1)');

% percentiles
p_knee = prctile(T_use.KneeMax(~isnan(T_use.KneeMax)), [1 5 10 25 50 75 90 95 99]);
p_rom  = prctile(T_use.ROM_swing(~isnan(T_use.ROM_swing)), [1 5 10 25 50 75 90 95 99]);
disp('KneeMax percentiles:'); disp(p_knee);
disp('ROM_swing percentiles:'); disp(p_rom);

% vGRF distribution
figure; histogram(T_use.vGRFmax_norm(~isnan(T_use.vGRFmax_norm)),50);
xlabel('vGRFmax (BW-normalized)'); title('Peak vGRF distribution (use this = 1)');
