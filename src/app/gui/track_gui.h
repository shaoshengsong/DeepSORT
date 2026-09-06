#pragma once

// DeepTracker GUI - control IDs and message definitions
// Core tracking algorithm by: shaoshengsong

#define IDD_MAIN        100
#define IDC_VIDEO_EDIT  1001
#define IDC_VIDEO_BTN   1002
#define IDC_OUT_EDIT    1003
#define IDC_OUT_BTN     1004
#define IDC_TRACK_CHK   1005
#define IDC_CONF_EDIT   1006
#define IDC_CLASS_EDIT  1007
#define IDC_PREVIEW_CHK 1008
#define IDC_RUN_BTN     1009
#define IDC_STOP_BTN    1010
#define IDC_PROGRESS    1011
#define IDC_LOG         1012
#define IDC_ABOUT       1013

// Custom messages: worker thread -> dialog
#define WM_GUI_LOG  (WM_APP + 0)   // lParam: char* (must be freed)
#define WM_GUI_PROG (WM_APP + 1)   // lParam: char* (must be freed)
#define WM_GUI_DONE (WM_APP + 2)   // wParam: 0=OK/cancel, 1=error
