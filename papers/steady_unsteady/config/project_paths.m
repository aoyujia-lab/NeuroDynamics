function P = project_paths()
% PROJECT_PATHS
% Centralized path configuration for the Steady–Unsteady project.
%
% This function defines all code, data, and output directories used in the
% paper. It should be called once at the beginning of any analysis script:
%
%   P = project_paths();

%% ==============================================================
%  Code root (repository root)
%  ==============================================================

% Locate code root by going up three levels from this file
code_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));

P.code.root        = code_root;
P.code.commonfuncs = fullfile(code_root, 'common_funcs');

paper_name = 'steady_unsteady';
P.code.analysisfuncs  = fullfile(code_root, paper_name, 'funcs\analysis');
ensure_dir(P.code.analysisfuncs);
P.code.preprocessfuncs  = fullfile(code_root, paper_name, 'funcs\preprocess');
ensure_dir(P.code.preprocessfuncs);

%% ==============================================================
%  Data root (external to code)
%  ==============================================================

% ---- MODIFY THIS PATH IF DATA LOCATION CHANGES ----
data_root = 'E:\DATA\Steady-unsteady'; 

P.data.root            = data_root;
P.data.raw             = fullfile(data_root, 'raw');
P.data.roisignals      = fullfile(data_root, 'roisignals_gsr');
P.data.behav           = fullfile(data_root, 'Results', 'Behavior');

%% ==============================================================
%  Output / results
%  ==============================================================

P.results.root  = fullfile(code_root, paper_name, 'results');
P.results.cache = fullfile(P.results.root, 'cache');    % intermediate results
P.results.fig   = fullfile(P.results.root, 'figures');  % final figures

ensure_dir(P.results.root);
ensure_dir(P.results.cache);
ensure_dir(P.results.fig);

%% ==============================================================
%  addpath
%  ==============================================================
addpath(P.code.analysisfuncs)

end

%% ==============================================================
%  Local utility
%  ==============================================================

function ensure_dir(p)
%ENSURE_DIR Create directory if it does not exist
if ~exist(p, 'dir')
    mkdir(p);
end
end