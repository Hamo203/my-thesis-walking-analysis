%% === 設定 ===
%%11\5時点で最新
%%vGRF処理とtrc->遊脚期における膝関節角度の算出
clear; clc;

trcDir = "rightfootdata"; % .trcフォルダ
motDir = trcDir;
outCSV = fullfile(trcDir, "summary_knee_vGRF_QC_kai_ver4.csv");

fs_trc = 200;       % トラッキングデータ 200Hz
cutoff_trc = 6;     % カットオフ周波数 for 膝
fs_mot = 1000;      % 床反力計 1000Hz
cutoff_mot = 10;    % カットオフ周波数 for vGRF
BW_default = 70;    % デフォルト体重[kg]

files = dir(fullfile(trcDir, "*.trc"));
results = {};

fprintf("Processing %d files...\n", numel(files));

%% === 各ファイル処理 ===
for f = 1:numel(files)
    trcFile = fullfile(files(f).folder, files(f).name);
    [~, baseName, ~] = fileparts(trcFile);
    fprintf("\n=== %s ===\n", baseName);
    flag_RTO_mismatch = false;

    %% --- TRC読込 ---
    fid = fopen(trcFile, 'r');
    if fid < 0
        warning("Failed to open %s", trcFile);
        continue;
    end
    lines = cell(1,5);
    for i = 1:5
        lines{i} = fgetl(fid);
    end
    fclose(fid);

    headerMarkers = strsplit(strtrim(lines{4}), '\t');
    headerCoords  = strsplit(strtrim(lines{5}), '\t');
    headerMarkers = headerMarkers(~cellfun('isempty', headerMarkers));
    headerCoords  = headerCoords(~cellfun('isempty', headerCoords));

    headers = ["Frame#", "Time"];
    for i = 3:length(headerMarkers)
        m = matlab.lang.makeValidName(headerMarkers{i});
        headers = [headers, m + ".X", m + ".Y", m + ".Z"];
    end

    opts = detectImportOptions(trcFile, 'FileType','text');
    opts.DataLines = [6, Inf];
    opts.Delimiter = '\t';
    data = readmatrix(trcFile, opts);

    if numel(headers) ~= size(data,2)
        warning("Header列数(%d) ≠ Data列数(%d)。ズレ修正します。", numel(headers), size(data,2));
        headers = headers(1:min(numel(headers), size(data,2)));
    end
    T = array2table(data, 'VariableNames', headers);

    % --- マーカー抽出 ---
    req = ["RASI","LASI","SACR","RKNE","RKN2","RANK","RAN2"];
    idx = struct();
    for i = 1:numel(req)
        m = req(i);
        xcol = find(strcmpi(headers, m + ".X"));
        ycol = find(strcmpi(headers, m + ".Y"));
        zcol = find(strcmpi(headers, m + ".Z"));
        if isempty(xcol) || isempty(ycol) || isempty(zcol)
            warning("%s not found in %s", m, baseName);
            idx.(m) = NaN(1,3);
        else
            idx.(m) = [xcol, ycol, zcol];
        end
    end

    % --- 各マーカー座標 ---
    RASI = T{:, idx.RASI};
    LASI = T{:, idx.LASI};
    SACR = T{:, idx.SACR};
    RKNE = T{:, idx.RKNE};
    RKN2 = T{:, idx.RKN2};
    RANK = T{:, idx.RANK};
    RAN2 = T{:, idx.RAN2};

    % --- ローパスフィルタ ---
    [b_trc, a_trc] = butter(4, cutoff_trc / (fs_trc / 2), 'low');
    RASI = filtfilt(b_trc, a_trc, RASI);
    LASI = filtfilt(b_trc, a_trc, LASI);
    SACR = filtfilt(b_trc, a_trc, SACR);
    RKNE = filtfilt(b_trc, a_trc, RKNE);
    RKN2 = filtfilt(b_trc, a_trc, RKN2);
    RANK = filtfilt(b_trc, a_trc, RANK);
    RAN2 = filtfilt(b_trc, a_trc, RAN2);

    %% --- 膝角度算出 ---
    hip_center   = ((RASI + LASI)/2 + SACR) / 2;
    knee_center  = (RKNE + RKN2) / 2;
    ankle_center = (RANK + RAN2) / 2;

    v_thigh = hip_center - knee_center;
    v_shank = ankle_center - knee_center;
    dot_val = sum(v_thigh .* v_shank, 2);
    norm_th = sqrt(sum(v_thigh.^2,2));
    norm_sh = sqrt(sum(v_shank.^2,2));
    cosval = dot_val ./ (norm_th .* norm_sh);
    cosval = max(min(cosval, 1), -1);
    angle_between = acosd(cosval);

    knee_flex = 180 - angle_between;
    knee_flex(knee_flex < -10 | knee_flex > 200) = NaN;
    knee_angle=knee_flex

    knee_min = min(knee_flex);
    knee_max = max(knee_flex);
    rom_total = knee_max - knee_min;
    fprintf('Total ROM: %.2f deg\n', rom_total);

    %% --- イベント処理 ---
    eventFile = fullfile(trcDir, replace(baseName, "_markers", "_events.csv"));
    rom_swing = NaN;
    missing_event_reason = "";

    if isfile(eventFile)
        E = readtable(eventFile);
        rhs_time = E.Time_s_(strcmp(E.Label, 'RHS'));
        rto_time = E.Time_s_(strcmp(E.Label, 'RTO'));
        disp(rto_time)
        disp(rhs_time)
        

        if isempty(rhs_time) || isempty(rto_time)
            missing_event_reason = "missing_event";
            warning("イベント不足: %s", baseName);
        else
            found_pair = false;
            t = T.Time;

            for i = 1:numel(rto_time)
                rhs_after = rhs_time(rhs_time > rto_time(i));
                if isempty(rhs_after)
                    continue;
                end
                rhs_curr = rhs_after(1);

                [~, idx_RTO] = min(abs(t - rto_time(i)));
                [~, idx_RHS] = min(abs(t - rhs_curr));

                if idx_RHS > idx_RTO
                    % 区間（RTO -> RHS）を切り出す
                    knee_seg = knee_flex(idx_RTO:idx_RHS);
                    % RTO時の角度 
                    angle_at_RTO = knee_flex(idx_RTO);
                    % 区間内のピーク（最大屈曲）とそのインデックス ---
                    [local_peak_val, local_peak_idx_rel] = max(knee_seg);    % 区間内の最大値（相対インデックス）
                    idx_peak_local = idx_RTO - 1 + local_peak_idx_rel;       % 全体インデックスに変換
                    t_peak_local = T.Time(idx_peak_local);
                
                    

                    % RTO -> Peak の ROM を計算（peak - angle@RTO)
                    rom_to_peak = local_peak_val - angle_at_RTO;
                    if isnan(rom_to_peak) || rom_to_peak < 0
                        % 念のため負値やNaNは NaN にする（データ異常時の安全策）
                        rom_to_peak = NaN;
                    end
                    rom_swing = rom_to_peak;

                    fprintf('RTO→RHS 区間: %.3f→%.3f s, ROM(RTO->peak)=%.2f° (peak=%.2f°, RTO=%.2f°)\n', ...
                        rto_time(i), rhs_curr, rom_swing, local_peak_val, angle_at_RTO);
                    
                    % 区間の主要マーカー座標（先頭5行だけ表示）
                    disp('=== 各マーカー座標(先頭5行) ===');
                    n_show = min(5, idx_RHS - idx_RTO + 1);
                    disp(table( ...
                        T{idx_RTO:idx_RTO+n_show-1, idx.RASI}, ...
                        T{idx_RTO:idx_RTO+n_show-1, idx.LASI}, ...
                        T{idx_RTO:idx_RTO+n_show-1, idx.SACR}, ...
                        T{idx_RTO:idx_RTO+n_show-1, idx.RKNE}, ...
                        T{idx_RTO:idx_RTO+n_show-1, idx.RKN2}, ...
                        T{idx_RTO:idx_RTO+n_show-1, idx.RANK}, ...
                        T{idx_RTO:idx_RTO+n_show-1, idx.RAN2}, ...
                        'VariableNames', {'RASI','LASI','SACR','RKNE','RKN2','RANK','RAN2'} ...
                    ));
                
                    % 膝角度波形の確認（先頭10点だけ）
                    disp('=== 膝角度データ ===');
                    %disp(knee_seg(1:min(10, numel(knee_seg))));
                    disp(knee_seg)
                    % RTOのフレーム時間（TRC基準）
                    t_rto_frame = T.Time(idx_RTO);
                    % 判定閾値（秒）
                    peak_thresh = 0.05; % ±50 ms（必要なら調整）
                
                    flag_RTO_local_mismatch = false;
                    if abs(t_rto_frame - t_peak_local) < peak_thresh
                        flag_RTO_local_mismatch = true;
                        fprintf('local: RTO(fr %.0f, %.3f s) が区間内ピーク(%.3f s) と近接 ⇒ RTOずれ疑い\n', ...
                            idx_RTO, t_rto_frame, t_peak_local);
                    end

                    if flag_RTO_local_mismatch
                        flag_RTO_mismatch = true;
                    end

                    
                    found_pair = true;
                    break;
                end
            end

            if ~found_pair
                warning("RTO→RHSペアが見つからず: %s", baseName);
                missing_event_reason = "no_valid_pair";
            end
        end
    else
        warning("イベントCSVなし: %s", eventFile);
        missing_event_reason = "no_event_csv";
    end

    %% --- vGRF処理 ---
    motFile = fullfile(motDir, replace(baseName, "_markers", "_forces_COP") + ".mot");
    if ~isfile(motFile)
        warning("対応するmotが見つかりません: %s", motFile);
        vGRF_norm_max = NaN;
    else
        mot = readtable(motFile, 'FileType', 'text');
        vars = mot.Properties.VariableNames;
        vars_lower = lower(vars);
        isFy = contains(vars_lower, 'ground_force_') & contains(vars_lower, '_vy');
        fy_cols = vars(isFy);

        BW = BW_default;
        tokens = regexp(baseName, '[0-9]+', 'match');
        if ~isempty(tokens)
            nums = cellfun(@str2double, tokens);
            BW_candidates = nums(nums>=30 & nums<=200);
            if ~isempty(BW_candidates)
                BW = BW_candidates(end);
            end
        end

        if isempty(fy_cols)
            warning('垂直成分 (ground_force_*_vy) が見つかりません。');
            vGRF_norm_max = NaN;
        else
            vGRF_all = mot{:, fy_cols};
            [b,a] = butter(4, cutoff_mot / (fs_mot/2), 'low');
            vGRF_all_filt = filtfilt(b, a, vGRF_all);
            vGRF = max(vGRF_all_filt, [], 2);
            vGRF_norm = vGRF / (BW * 9.81);
            vGRF_norm_max = max(vGRF_norm);
        end
    end

    %% --- QC判定 ---

    qc_flags = {};
    nFrames = size(T,1);
    nan_frames = sum(any(isnan([RASI,LASI,SACR,RKNE,RKN2,RANK,RAN2]),2));
    if nan_frames > 0.05 * nFrames
        qc_flags{end+1} = sprintf('FLAG_NaN(%.1f%%)', 100 * nan_frames / nFrames);
    end
    if knee_max > 140 || knee_min < 0
        qc_flags{end+1} = sprintf('FLAG_angle_extreme(Kmax=%.1f Kmin=%.1f)', knee_max, knee_min);
    end
    if flag_RTO_mismatch
        qc_flags{end+1} = 'FLAG_RTO_mismatch';
    end
    if ~isnan(vGRF_norm_max)
        if vGRF_norm_max > 2.5 || vGRF_norm_max < 0.4
            qc_flags{end+1} = sprintf('FLAG_vGRF(%.2f)', vGRF_norm_max);
        end
    else
        qc_flags{end+1} = 'FLAG_vGRF_missing';
    end
    if ~isempty(missing_event_reason) && missing_event_reason ~= ""
        qc_flags{end+1} = sprintf('FLAG_event_missing(%s)', missing_event_reason);
    end

    use_this = isempty(qc_flags);
    use_reason = 'OK';
    if ~use_this, use_reason = strjoin(qc_flags, '; '); end

    %% --- 結果格納 ---
    results{end+1,1} = files(f).name;
    results{end,2} = knee_max;
    results{end,3} = knee_min;
    results{end,4} = rom_total;
    results{end,5} = rom_swing;
    results{end,6} = vGRF_norm_max;
    results{end,7} = use_this;
    results{end,8} = use_reason;
    results{end,9} = missing_event_reason;
end

%% === CSV出力 ===
T = cell2table(results, 'VariableNames', ...
    {'File','KneeMax','KneeMin','ROM_total','ROM_swing','vGRFmax_norm','use_this','use_reason','event_status'});
writetable(T, outCSV);
fprintf('\n 結果を保存しました: %s\n', outCSV);
