; Inno Setup script for ObjectRemover.
; Builds a SMALL installer: AI model weights are NOT bundled — they are
; downloaded once on first launch into %LOCALAPPDATA%\ObjectRemover\models.
;
; Build (after `pyinstaller packaging\object_remover.spec`):
;   iscc packaging\installer.iss

[Setup]
AppId={{8D2B1E44-6C77-4E0A-9C4F-2B7A5D9E1F30}
AppName=ObjectRemover
AppVersion=0.1.0
AppPublisher=ImageObectRemoval
DefaultDirName={autopf}\ObjectRemover
DefaultGroupName=ObjectRemover
OutputBaseFilename=ObjectRemover-Setup-0.1.0
Compression=lzma2/max
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64
WizardStyle=modern
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "..\dist\ObjectRemover\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\ObjectRemover"; Filename: "{app}\ObjectRemover.exe"
Name: "{autodesktop}\ObjectRemover"; Filename: "{app}\ObjectRemover.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\ObjectRemover.exe"; Description: "{cm:LaunchProgram,ObjectRemover}"; Flags: nowait postinstall skipifsilent
