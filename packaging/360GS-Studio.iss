#define MyAppName "360GS Studio"
#define MyAppVersion "0.1.0"
#define MyAppPublisher "360GS Studio contributors"
#define MyAppExeName "360GS Studio.exe"

[Setup]
AppId={{33A05815-D9BE-44D6-94C2-C4D654650257}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\360GS Studio
DefaultGroupName=360GS Studio
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
Compression=lzma2
SolidCompression=yes
OutputBaseFilename=360GS-Studio-{#MyAppVersion}-setup
WizardStyle=modern
PrivilegesRequired=lowest

[Files]
Source: "..\dist\360GS Studio\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\LICENSE"; DestDir: "{app}"
Source: "..\NOTICE.md"; DestDir: "{app}"
Source: "..\THIRD_PARTY_LICENSES.md"; DestDir: "{app}"

[Icons]
Name: "{group}\360GS Studio"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\360GS Studio"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch 360GS Studio"; Flags: nowait postinstall skipifsilent
