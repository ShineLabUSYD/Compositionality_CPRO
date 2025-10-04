%% Extract timeseries for instruction period

% Load storage.mat, sub_id2.mat

% For every participant
groupI_ts = [];
all_names = {};
a = 1;
for ii = 1:length(sub_id2)
    sub_ts = [];
    % Cycle through storage to get each instruction period
    for jj = 1:length(storage)
        name = storage(jj).name;
        if contains(name,sub_id2(ii))
            dm = storage(jj).conv_dmInstruct;
            ts = storage(jj).ts(:,1:482);
            all_names{a,1} = name;
            a = a + 1;
            % Cycle through each column (task ID) and take columns with values > 0
            for col = 1:size(dm,2)
                trial = dm(:,col)~=0;
                % Shorten to consistent length (~40 frames) - add to correct row of
                % collated timeseries
                if sum(trial,1)>0
                    tsT = ts(trial,:);
                    tsT = tsT(1:36,:); % Minimum frames for all mini-blocks
                    tsT_rs = reshape(tsT,[],1)';
                    if isempty(sub_ts) || col>size(sub_ts,1)
                        sub_ts(col,:) = tsT_rs;
                    elseif sum(sub_ts(col,:))==0
                        sub_ts(col,:) = tsT_rs;
                    else
                        sub_ts(64+col,:) = tsT_rs;
                    end
                end
            end
        end
        disp([ii jj]);
    end
    groupI_ts = cat(3,groupI_ts,sub_ts);
end


%% Extract only timeseries for trials

% Load storage.mat, sub_id2.mat

% For every subject
groupT_ts = [];
all_names = {};
a = 1;
for ii = 1:length(sub_id2)
    sub_ts = [];
    % Cycle through storage - For each scan take conv_dm
    for jj = 1:length(storage)
        name = storage(jj).name;
        if contains(name,sub_id2(ii))
            ts = storage(jj).ts(:,1:482);
            dm = storage(jj).conv_dm;
            all_names{a,1} = name;
            a = a + 1;
            % Cycle through each column (task ID) and take columns with values > 0
            for col = 1:size(dm,2)
                trial = dm(:,col)~=0;
                % Shorten to consistent length (~40 frames) - add to correct row of
                % collated timeseries
                if sum(trial,1)>0
                    tsT = ts(trial,:);
                    tsT = tsT(1:23,:); % Minimum 23 frames for all mini-blocks
                    tsT_rs = reshape(tsT,[],1)';
                    if isempty(sub_ts) || col>size(sub_ts,1)
                        sub_ts(col,:) = tsT_rs;
                    elseif sum(sub_ts(col,:))==0
                        sub_ts(col,:) = tsT_rs;
                    else
                        sub_ts(64+col,:) = tsT_rs;
                    end
                end
            end
        end
        disp([ii jj]);
    end
    groupT_ts = cat(3,groupT_ts,sub_ts); % Task ID*2 X Time*ROI X Participant
end


%% Partial Least Squares (split-half test reliability)
% Mean-centred task correlation PLS

% Load in coef_FIR.mat, taskset_ord.mat
ntask = 64;
nroi = 482;
nsub = 87;

% Using coef_FIR.mat
ntime = 23;
X = permute(reshape(coef_FIR,ntime,[],nroi,nsub),[2 4 1 3]); % Task ID X nsub X ntime X nroi
X1 = X(:,1:44,:,:); % First half
X2 = X(:,45:end,:,:); % Second half
X1_rs = reshape(X1,[],nroi*ntime); % Task ID*Participant X Time*ROI
X2_rs = reshape(X2,[],nroi*ntime);

Y1 = repmat(taskset_ord(:,1:3),size(X1,2),1); % Task ID*Participant
Y2 = repmat(taskset_ord(:,1:3),size(X2,2),1); % Task ID*Participant
% One-hot coding - All rules
Y = Y2;
Y_rules = zeros(size(Y,1),12);
Y_rules(Y(:,1)==1,1) = 1;
Y_rules(Y(:,1)==2,2) = 1;
Y_rules(Y(:,1)==3,3) = 1;
Y_rules(Y(:,1)==4,4) = 1;
Y_rules(Y(:,2)==1,5) = 1;
Y_rules(Y(:,2)==2,6) = 1;
Y_rules(Y(:,2)==3,7) = 1;
Y_rules(Y(:,2)==4,8) = 1;
Y_rules(Y(:,3)==1,9) = 1;
Y_rules(Y(:,3)==2,10) = 1;
Y_rules(Y(:,3)==3,11) = 1;
Y_rules(Y(:,3)==4,12) = 1;

Y = Y_rules;
X = X2_rs;
% Swapping labels
% Y(Y2(:,3)==2,3) = 4;
% Y(Y2(:,3)==4,3) = 2;

M = (diag(sum(Y,1))^-1)*Y'*X;

% Mean-centre M
M_mean = mean(M,1);
R = M - M_mean;

[U,S,V] = svd(R,'econ'); % U = Y, V = X, S = Eigenvalues
eig_val = real(diag(S)); % Get eigenvalues from diagonal
[eig_sort,idx] = sort(eig_val,'descend'); % Eigenvalues already sorted correctly
% Calculate explained variance
exp_var = 100.*diag(S).^2./sum(diag(S).^2);
figure;
p = plot(exp_var,'-','LineWidth',3,'Color','black');
p.Marker = '.';
p.MarkerFaceColor = 'black';
p.MarkerSize = 30;
%yline(1,'LineWidth',1.5,'Color','red');
%xlabel('Latent Variables');
%ylabel('Explained Variance');
set(gca,'box','off','FontSize',24,'FontName','Arial','linew',1.5,'TickDir','out');

% Accumulative explained variance
accum_exp_var = cumsum(exp_var);
figure;
p = plot(accum_exp_var,'-','LineWidth',3,'Color','black');
p.Marker = '.';
p.MarkerFaceColor = 'black';
p.MarkerSize = 30;
yline(90,'LineWidth',3,'Color','red','LineStyle','--');
%xlabel('Latent Variables');
%ylabel('Cumulative Explained Variance');
set(gca,'box','off','FontSize',24,'FontName','Arial','linew',1.5,'TickDir','out');

% Permutation testing for latent variables - significance against null
% distributions (noise)
% Randomly reorder rows of X, Y is unchanged
% Collect singular values for each iteration (i.e., 1000) = sampling
% distribution for null hypothesis
perm_idx = [];
for ii = 1:1000
    perm_idx(:,ii) = randperm(size(X,1));
end
% Permutation testing - Group-level cross-product
S_perm = [];
for jj = 1:size(perm_idx,2)
    idx = perm_idx(:,jj);
    X2 = X(idx,:);
    M2 = (diag(sum(Y,1))^-1)*Y'*X2;
    M2_mean = mean(M2,1);
    R2 = M2 - M2_mean;
    [Un,Sn,Vn] = svd(R2,'econ');
    S_perm(:,jj) = diag(Sn);
    disp(jj);
end

% Compare singular values for each latent observed against null distribution
pval = [];
iter = size(S_perm,2);
for ii = 1:length(eig_val)
    obs_val = eig_val(ii);
    null = S_perm(ii,:);
    pos_pval = sum(null >= obs_val)/iter;
    neg_pval = sum(null <= obs_val)/iter;
    %pval(ii,1) = 2*min([pos_pval neg_pval]);
    %pval(ii,1) = 2*pos_pval;

    % One-tailed test (conservative and avoids p = 0)
    pval(ii,1) = (sum(null >= obs_val) + 1) / (iter + 1);
end

% Check components of task contrast

% Motor Rules
data2 = U(:,1);
figure; 
bm = bar(data2,0.8,'FaceColor','flat','EdgeColor','black','LineWidth',1);
bm.CData(1:8,:) = repmat([236 236 236]./255,8,1);
bm.CData(9,:) = [0 149 107]./255;
bm.CData(10,:) = [0 171 123]./255;
bm.CData(11,:) = [0 195 138]./255;
bm.CData(12,:) = [0 226 159]./255;
set(gca,'box','off','FontSize',24,'FontName','Arial','linew',1.5,'TickDir','out', ...
    'XTickLabel',[]);

% Logic Rules
data2 = U(:,2);
figure; 
bl = bar(data2,0.8,'FaceColor','flat','EdgeColor','black','LineWidth',1);
bl.CData(1,:) = [129 75 126]./255;
bl.CData(2,:) = [212 151 207]./255;
bl.CData(3,:) = [159 80 147]./255;
bl.CData(4,:) = [198 115 192]./255;
bl.CData(5:12,:) = repmat([236 236 236]./255,8,1);
set(gca,'box','off','FontSize',24,'FontName','Arial','linew',1.5,'TickDir','out', ...
    'XTickLabel',[]);

% Sensory Rules
data2 = U(:,3);
figure; 
bs = bar(data2,0.8,'FaceColor','flat','EdgeColor','black','LineWidth',1);
bs.CData(5,:) = [214 122 0]./255;
bs.CData(6,:) = [255 151 32]./255;
bs.CData(7,:) = [255 172 88]./255;
bs.CData(8,:) = [255 188 170]./255;
bs.CData(1:4,:) = repmat([236 236 236]./255,4,1);
bs.CData(9:12,:) = repmat([236 236 236]./255,4,1);
set(gca,'box','off','FontSize',24,'FontName','Arial','linew',1.5,'TickDir','out', ...
    'XTickLabel',[]);

% V = voxel-dependent differences in brain-behaviour correlation
% Explore brain spatial maps
% V = Time*ROI*2 X 3
ntime = 23;
V_rs = reshape(V,ntime,nroi,size(V,2));

% Visualise average brain loading
V_mean = squeeze(mean(V_rs,1));
data = V_mean(:,1);
limits = [min(data) max(data)];
surf_schaef2(data(1:400),limits);
surf_cbm(data(455:482),limits);
subcort_plot(data); colormap(custom());

% Rerun above code for 1st and 2nd half of data
V_first = V_mean;
V_second = V_mean;
% Compare spatial vectors between first and second half
[V_corr,V_pval] = corr(V_first,V_second);
% V1: r = 0.9862, p = 0
% V2: r = 0.9727, p = 0
% V3: r = 0.8718, p = 0


%% Partial Least Squares (Mean-centred Task)
% Using all data

ntask = 64;
nroi = 482;
nsub = 87;

% Using coef_FIR.mat
ntime = 23;
X = permute(reshape(coef_FIR,ntime,[],nroi,nsub),[2 4 1 3]); % Task ID X nsub X ntime X nroi
X = reshape(X,[],nroi*ntime); % Task ID*Participant X Time*ROI

Y = repmat(taskset_ord(:,1:3),size(X,2),1); % Task ID*Participant
% One-hot coding - All rules
Y2 = Y;
Y_rules = zeros(size(Y,1),12);
Y_rules(Y2(:,1)==1,1) = 1;
Y_rules(Y2(:,1)==2,2) = 1;
Y_rules(Y2(:,1)==3,3) = 1;
Y_rules(Y2(:,1)==4,4) = 1;
Y_rules(Y2(:,2)==1,5) = 1;
Y_rules(Y2(:,2)==2,6) = 1;
Y_rules(Y2(:,2)==3,7) = 1;
Y_rules(Y2(:,2)==4,8) = 1;
Y_rules(Y2(:,3)==1,9) = 1;
Y_rules(Y2(:,3)==2,10) = 1;
Y_rules(Y2(:,3)==3,11) = 1;
Y_rules(Y2(:,3)==4,12) = 1;

Y = Y_rules;
% Swapping labels
% Y(Y2(:,3)==2,3) = 4;
% Y(Y2(:,3)==4,3) = 2;

M = (diag(sum(Y,1))^-1)*Y'*X;

% Mean-centre M
M_mean = mean(M,1);
R = M - M_mean;

[U,S,V] = svd(R,'econ'); % U = Y, V = X, S = Eigenvalues
eig_val = real(diag(S)); % Get eigenvalues from diagonal
[eig_sort,idx] = sort(eig_val,'descend'); % Eigenvalues already sorted correctly
% Calculate explained variance
exp_var = 100.*diag(S).^2./sum(diag(S).^2);
figure;
p = plot(exp_var,'-','LineWidth',3,'Color','black');
p.Marker = '.';
p.MarkerFaceColor = 'black';
p.MarkerSize = 30;
%yline(1,'LineWidth',1.5,'Color','red');
%xlabel('Latent Variables');
%ylabel('Explained Variance');
set(gca,'box','off','FontSize',24,'FontName','Arial','linew',1.5,'TickDir','out');

% Accumulative explained variance
accum_exp_var = cumsum(exp_var);
figure;
p = plot(accum_exp_var,'-','LineWidth',3,'Color','black');
p.Marker = '.';
p.MarkerFaceColor = 'black';
p.MarkerSize = 30;
yline(90,'LineWidth',3,'Color','red','LineStyle','--');
%xlabel('Latent Variables');
%ylabel('Cumulative Explained Variance');
set(gca,'box','off','FontSize',24,'FontName','Arial','linew',1.5,'TickDir','out');

% Check components of task contrast

% Motor Rules
data2 = U(:,1);
figure; 
bm = bar(data2,0.8,'FaceColor','flat','EdgeColor','black','LineWidth',1);
bm.CData(1:8,:) = repmat([236 236 236]./255,8,1);
bm.CData(9,:) = [0 149 107]./255;
bm.CData(10,:) = [0 171 123]./255;
bm.CData(11,:) = [0 195 138]./255;
bm.CData(12,:) = [0 226 159]./255;
set(gca,'box','off','FontSize',24,'FontName','Arial','linew',1.5,'TickDir','out', ...
    'XTickLabel',[]);

% Logic Rules
data2 = U(:,2);
figure; 
bl = bar(data2,0.8,'FaceColor','flat','EdgeColor','black','LineWidth',1);
bl.CData(1,:) = [129 75 126]./255;
bl.CData(2,:) = [212 151 207]./255;
bl.CData(3,:) = [159 80 147]./255;
bl.CData(4,:) = [198 115 192]./255;
bl.CData(5:12,:) = repmat([236 236 236]./255,8,1);
set(gca,'box','off','FontSize',24,'FontName','Arial','linew',1.5,'TickDir','out', ...
    'XTickLabel',[]);

% Sensory Rules
data2 = U(:,3);
figure; 
bs = bar(data2,0.8,'FaceColor','flat','EdgeColor','black','LineWidth',1);
bs.CData(5,:) = [214 122 0]./255;
bs.CData(6,:) = [255 151 32]./255;
bs.CData(7,:) = [255 172 88]./255;
bs.CData(8,:) = [255 188 170]./255;
bs.CData(1:4,:) = repmat([236 236 236]./255,4,1);
bs.CData(9:12,:) = repmat([236 236 236]./255,4,1);
set(gca,'box','off','FontSize',24,'FontName','Arial','linew',1.5,'TickDir','out', ...
    'XTickLabel',[]);

% V = voxel-dependent differences in brain-behaviour correlation
% Explore brain spatial maps
% V = Time*ROI*2 X 3
ntime = 23;
V_rs = reshape(V,ntime,nroi,size(V,2));

% Visualise average brain loading
V_mean = squeeze(mean(V_rs,1));
data = V_mean(:,1);
limits = [min(data) max(data)];
surf_schaef2(data(1:400),limits);
surf_cbm(data(455:482),limits);
subcort_plot(data); colormap(custom());

% Validate with Instruction period
% Rerun PLS on groupI_ts.mat
ntime = 36; 
V_rs2 = reshape(V2,ntime,nroi,size(V,2));
V_mean2 = squeeze(mean(V_rs,1));
% Correlate brain patterns of Instruction vs trial period
[r,pval] = corr(V_mean(:,1),V_mean2(:,1)); % r = -0.9933, p = 0
[r,pval] = corr(V_mean(:,2),V_mean2(:,2)); % r = -0.9838, p = 0
[r,pval] = corr(V_mean(:,3),V_mean2(:,3)); % r = -0.9596, p = 0

% Visualise static average
figure; 
scatter3(V_mean(:,1),V_mean(:,2),V_mean(:,3),30,'filled');
set(gca,'box','off','FontSize',24,'FontName','Arial','linew',1.5,'TickDir','out');

% Assign regions to each domain
V_abs = abs(V_mean(:,1:3));
% Assign region based on largest value
[~,domain_id] = max(V_abs,[],2);
domain_id2 = domain_id;
for ii = 1:size(domain_id,1)
    max_val = V_mean(ii,domain_id(ii));
    if max_val<0
        domain_id2(ii) = domain_id(ii)+0.5;
    end
end


%% Permutation testing for latent variables - significance against null distributions (noise)
% Randomly reorder rows of X, Y is unchanged
% Collect singular values for each iteration (i.e., 1000) = sampling
% distribution for null hypothesis
perm_idx = [];
for ii = 1:1000
    perm_idx(:,ii) = randperm(size(X,1));
end
% Permutation testing - Group-level cross-product
S_perm = [];
for jj = 1:size(perm_idx,2)
    idx = perm_idx(:,jj);
    X2 = X(idx,:);
    M2 = (diag(sum(Y,1))^-1)*Y'*X2;
    M2_mean = mean(M2,1);
    R2 = M2 - M2_mean;
    [Un,Sn,Vn] = svd(R2,'econ');
    S_perm(:,jj) = diag(Sn);
    disp(jj);
end

% Compare singular values for each latent observed against null distribution
pval = [];
iter = size(S_perm,2);
for ii = 1:length(eig_val)
    obs_val = eig_val(ii);
    null = S_perm(ii,:);
    pos_pval = sum(null >= obs_val)/iter;
    neg_pval = sum(null <= obs_val)/iter;
    %pval(ii,1) = 2*min([pos_pval neg_pval]);
    %pval(ii,1) = 2*pos_pval;

    % One-tailed test (conservative and avoids p = 0)
    pval(ii,1) = (sum(null >= obs_val) + 1) / (iter + 1);
end


%% Bootstrapping - confidence intervals (effect distributions), stability of
% latent variable through resampling
% Save mean_Vstab.mat to be used in bootstrap.mat script

% Stability evaluted by dividing latent variable by SE (values > 2 is stable)
% Sample with replacement observations in X and Y --> estimate SE from 1000
% samples --> estimate stability

% Inputs
% X: Nrows x P
% Y: Nrows x Q
% subj_id: Nrows x 1 labels (1..S) or arbitrary IDs
% B: number of bootstrap samples
% orig_salience: P x 1 (original salience for LV of interest)
subj_id = repelem([1:87]',64,1);
S = numel(unique(subj_id));          % number of subjects
subjects = unique(subj_id);          % unique subject IDs
V_boot = zeros(size(X,2),size(Y,2),1000);
rng(0);
for b = 1:1000
    % sample subjects with replacement
    sampled_subjects = subjects(randsample(S, S, true)); % S picks with replacement
    
    % build index vector of rows to include (concatenate all rows from each selected subject)
    rows_to_use = [];
    for s = 1:S
        sid = sampled_subjects(s);
        rows_s = find(subj_id == sid);   % all rows for that subject
        rows_to_use = [rows_to_use; rows_s];
    end
    
    Xb = X(rows_to_use, :);
    Yb = Y(rows_to_use, :);

    % PLS
    Mb = (diag(sum(Yb,1))^-1)*Yb'*Xb;
    Mb_mean = mean(Mb,1);
    Rb = Mb - Mb_mean;
    [Ub,Sb,Vb] = svd(Rb,'econ');

    % Align sign to original
    for l = 1:size(Vb,2)
        if dot(V(:,l), Vb(:,l)) < 0
            Vb(:,l) = -Vb(:,l);
            Ub(:,l) = -Ub(:,l);
        end
    end

    V_boot(:,:,b) = Vb;
    disp(b);
end

% Calculate SE for each region per latent variable
SE_boot = [];
for ii = 1:size(V_boot,2)
    VL = squeeze(V_boot(:,ii,:));
    SE_boot(:,ii) = std(VL,[],2)./sqrt(size(VL,2));
    disp(ii);
end
% Calculate stability for each region per latent variable
V_stab = V./SE_boot;

ntime = 23;
nroi = 482;
V_stab_rs = reshape(V_stab,ntime,nroi,[]);
mean_Vstab = squeeze(mean(V_stab_rs,1));
limits = [min(mean_Vstab,[],1)' max(mean_Vstab,[],1)'];
surf_schaef2(mean_Vstab(1:400,1),limits(1,:));
surf_cbm(mean_Vstab(455:482,1),limits(1,:));
subcort_plot(mean_Vstab(:,1),limits(1,:)); colormap(custom());


%% Figure 3a

% Plot single timepoint for example
V_t = squeeze(V_rs(7,:,:));
% Force high-quality rendering
opengl software;
% 3D scatter plot
% Load in roi_gen, roi_spec, domain_id2
% Colormap Face
c_pink = [255 28 99]./255;
c_silver = [97 127 150]./255;
c_grey = [241 241 241]./255;
c_motor = [0 149 36]./255;
c_logic = [129 33 126]./255;
c_sensory = [214 122 0]./255;
cmap_face = repmat(c_grey,482,1);
cmap_face(roi_gen==1,:) = repmat(c_pink,sum(roi_gen==1),1); % 65
%cmap_face(roi_spec==1,:) = repmat(c_silver,sum(roi_spec==1),1); % 54
cmap_face(domain_id2<2 & roi_spec==1,:) = repmat(c_motor,sum(domain_id2<2 & roi_spec==1),1);
cmap_face(domain_id2<3 & domain_id2>1.5 & roi_spec==1,:) = repmat(c_logic,sum(domain_id2<3 & domain_id2>1.5 & roi_spec==1),1);
cmap_face(domain_id2>2.5 & roi_spec==1,:) = repmat(c_sensory,sum(domain_id2>2.5 & roi_spec==1),1);
figure;
scatter3(V_t(:,1),-1.*V_t(:,2),V_t(:,3),50,cmap_face,'filled','MarkerFaceColor','flat', ...
    'MarkerFaceAlpha',0.7,'MarkerEdgeAlpha',1,'MarkerEdgeColor','flat');
% xlabel('Motor Domain');
% ylabel('Logic Domain');
% zlabel('Sensory Domain');
hold on
grid off; 
axis equal;
% Get axis limits
xL = xlim;
yL = ylim;
zL = zlim;
% Plot central axes
plot3([xL(1) xL(2)], [0 0], [0 0],'--', 'Color',[0 0 0 0.5], 'LineWidth', 1.5); % X-axis
plot3([0 0], [yL(1) yL(2)], [0 0], '--', 'Color', [0 0 0 0.5], 'LineWidth', 1.5); % Y-axis
plot3([0 0], [0 0], [zL(1) zL(2)], '--', 'Color', [0 0 0 0.5], 'LineWidth', 1.5); % Z-axis
set(gca,'Box','off','XColor','none','YColor','none','ZColor','none');
view(az_el);
hold off;

% Colormap edge
cmap_edge = repmat(c_grey,482,1);
cmap_edge(roi_gen==1,:) = repmat(c_pink,sum(roi_gen==1),1); % 65
cmap_edge(domain_id2<2 & roi_spec==1,:) = repmat(c_motor,sum(domain_id2<2 & roi_spec==1),1);
cmap_edge(domain_id2<3 & domain_id2>1.5 & roi_spec==1,:) = repmat(c_logic,sum(domain_id2<3 & domain_id2>1.5 & roi_spec==1),1);
cmap_edge(domain_id2>2.5 & roi_spec==1,:) = repmat(c_sensory,sum(domain_id2>2.5 & roi_spec==1),1);


%% Figure 3c-e

% Plot with joint dots for significant delta correlation
V_t = squeeze(V_rs(7,:,:));
V_t(:,2) = -1.*V_t(:,2);
V_tsample = V_t(roi_gen==1 | roi_spec==1,:);
% 3D scatter plot
% Load in roi_gen, roi_spec, domain_id2
% Colormap Face
c_pink = [255 28 99]./255;
c_silver = [97 127 150]./255;
c_grey = [255 255 255]./255;
c_motor = [0 149 36]./255;
c_logic = [129 33 126]./255;
c_sensory = [214 122 0]./255;
cmap_face = repmat(c_grey,482,1);
cmap_face(roi_gen==1,:) = repmat(c_pink,sum(roi_gen==1),1); % 65
%cmap_face(roi_spec==1,:) = repmat(c_silver,sum(roi_spec==1),1); % 54
cmap_face(domain_id2<2 & roi_spec==1,:) = repmat(c_motor,sum(domain_id2<2 & roi_spec==1),1);
cmap_face(domain_id2<3 & domain_id2>1.5 & roi_spec==1,:) = repmat(c_logic,sum(domain_id2<3 & domain_id2>1.5 & roi_spec==1),1);
cmap_face(domain_id2>2.5 & roi_spec==1,:) = repmat(c_sensory,sum(domain_id2>2.5 & roi_spec==1),1);
cmap_sample = cmap_face(roi_gen==1 | roi_spec==1,:);
% Force high-quality rendering
opengl software;
figure;
scatter3(V_tsample(:,1),V_tsample(:,2),V_tsample(:,3),50,cmap_sample,'filled','MarkerFaceColor','flat', ...
    'MarkerFaceAlpha',0.7,'MarkerEdgeAlpha',1,'MarkerEdgeColor','flat');
% xlabel('Motor Domain');
% ylabel('Logic Domain');
% zlabel('Sensory Domain');
hold on

% Load in hl, hm, hs
large_idx = find(roi_gen);
small_idx = find(hl);
src_idx = large_idx(small_idx);
large_idx2 = find(roi_spec);
small_idx2 = find(spec_domain==2.5);
target_idx = large_idx2(small_idx2);
for s = 1:length(src_idx)
    src = src_idx(s);
    tgt = target_idx;
    for i = 1:length(tgt)
        pt1 = V_t(src, :);
        pt2 = V_t(tgt(i), :);
        plot3([pt1(1) pt2(1)], [pt1(2) pt2(2)], [pt1(3) pt2(3)], ...
              '-', 'Color', [c_logic 0.5], 'LineWidth', 0.5);
    end
end

grid off; 
axis equal;
% Get axis limits
xL = xlim;
yL = ylim;
zL = zlim;
% Plot central axes
plot3([xL(1) xL(2)+0.01], [0 0], [0 0],'--', 'Color',[0 0 0 0.5], 'LineWidth', 1.5); % X-axis
plot3([0 0], [yL(1)-0.02 yL(2)+0.02], [0 0], '--', 'Color', [0 0 0 0.5], 'LineWidth', 1.5); % Y-axis
plot3([0 0], [0 0], [zL(1)-0.01 zL(2)+0.01], '--', 'Color', [0 0 0 0.5], 'LineWidth', 1.5); % Z-axis
set(gca,'Box','off','XColor','none','YColor','none','ZColor','none');
view(az_el);
hold off;


%% Figure 2g-i

% Plot individual axis
x = 1:23; % X values
y = V_rs(:,:,2); % Example function

cmap = colormap(cmapP); % Choose a colormap
cmin = min(y);
cmax = max(y);
numColors = size(cmap,1);

z = V_mean(:,2);
z_norm = (z - min(z)) / (max(z) - min(z));
colorIdx = round(z_norm * (numColors - 1)) + 1; % Get colormap indices
colorIdx = min(max(colorIdx, 1), numColors); % Ensure indices are within bounds
lineColors = cmap(colorIdx, :); % Assign colors
hold on
for ii = 1:size(y,2)
    plot(x, y(:, ii), 'Color', lineColors(ii, :), 'LineWidth', 3); % Assign color per line
end
yline(0,'--','Color',[0 0 0],'LineWidth',3);
hold off;
set(gca,'box','off','FontSize',24, 'FontName', 'Arial','linew',1.5,'TickDir','out');
axis('tight');
ylim([-0.05 0.05]);

hold on
plot(trial_conv./15,'LineStyle','--','LineWidth',2,'Color','red');
set(gca,'box','off','FontSize',24,'FontName','Arial','linew',1.5,'TickDir','out');
hAxes = findobj(gcf,'Type','axes');
hAxes.Position = [0.1300 0.1100 0.5057 0.8028];


%% Identifying rule-dependent and rule-indepedent regions

% Calculate distance from centre for each time point
centroid = [0 0 0];
dist_roi = [];
for ii = 1:size(V_rs,1)
    V_t = squeeze(V_rs(ii,:,1:3));
    dist_roi(:,ii) = pdist2(V_t,centroid,'euclidean');
end
% Plot regions distance from centroid across time
% Load in voltron_id2.mat
networks = voltron_id2(1:482);
networks(401:454) = 9;
cmapNetwork = [];
for ii = 1:length(unique(networks))
    network_id = networks==ii;
    num_rois = sum(network_id);
    cmapNetwork = [cmapNetwork; repmat(pastel_colormap(ii,:),num_rois,1)];
end
figure; 
plot(dist_roi','LineWidth',1.5);
set(gca,'box','off','FontSize',24, 'FontName', 'Arial','linew',1.5,'TickDir','out');
axis('tight');
xlim([1 23]);
ylim([0 0.1])
ax = gca;
ax.ColorOrder = cmapNetwork;

% Area under the curve for total specificity
area_distroi = cumtrapz(dist_roi');
total_area = area_distroi(end,:)';

% Plot z-scored specialisation
total_areaz = zscore(total_area,[],1);
%mean_distroiz = zscore(mean_distroi,[],1);
figure;
scatter(1:482,total_areaz,30,voltron_id2(1:482),'filled');
% Regions >1 SD from mean = specialised
figure; histogram(total_area);

% Plot mean coef_FIR
ntime = 23;
nroi = 482;
nsub = 87;
coef_FIRrs = reshape(coef_FIR,ntime,[],nroi,nsub);
mean_coefFIR = squeeze(mean(coef_FIRrs,[2 4]));
figure; 
plot(mean_coefFIR,'LineWidth',1.5);
yline(0,'--','Color',[0 0 0],'LineWidth',1.5);
set(gca,'box','off','FontSize',24, 'FontName', 'Arial','linew',1.5,'TickDir','out');
axis('tight');
xlim([1 23]);
ylim([-1.5 1.5]);
ax = gca;
ax.ColorOrder = cmapNetwork;


%% Figure 3b
% Z-score mean_coefFIR and compare to specialisation
% Transparent version
mean_coefFIR2 = squeeze(mean(coef_FIR,[1 3]))';
mean_coefFIRz = zscore(mean_coefFIR2,[],1);
figure; 
s = scatter(total_areaz,mean_coefFIRz,50,cmapNetwork,'MarkerFaceColor','flat', ...
    'MarkerFaceAlpha',0.7,'MarkerEdgeColor','flat');
set(gca,'box','off','FontSize',24,'FontName','Arial','linew',1.5,'TickDir','out');
xline(1,'--','LineWidth',3,'Color',[107 147 181]./255);
hold on
plot([-2;1],[1;1],'--','LineWidth',3,'Color',[212 160 0]./255);


%% Make IDs for rule-dependent (spec) and rule-indepedent (gen)

% Figure 3f
roi_spec = zeros(482,1);
roi_spec(total_areaz>1) = 1; % 54
limits = [0 1];
surf_schaef2(roi_spec(1:400),limits);
surf_cbm(roi_spec(455:482),limits);
subcort_plot(roi_spec); colormap(custom());

roi_gen = zeros(482,1);
roi_gen(total_areaz<=1 & mean_coefFIRz>1) = 1; % 65
% limits = [0 1];
% surf_schaef2(roi_gen(1:400),limits);
% surf_cbm(roi_gen(455:482),limits);
% subcort_plot(roi_gen); colormap(custom());
