
write-host "`r`nTNT installation for Windows`r`n"

if ( 0 -eq $($args.count) )
{
    $TNT_PATH="C:\Program Files\TNT"
    write-host "`r`nNo PATH specified for TNT. Selecting default PATH: $($TNT_PATH)`r`n"
}
else
{
    $TNT_PATH=$args[0]
    write-host "`r`nSpecified TNT PATH: $($TNT_PATH)`r`n"
}

mkdir $TNT_PATH

write-host "`r`nPreparing installation"

write-host "`r`nDownloading TNT"
$client = New-Object System.Net.WebClient
$client.DownloadFile('http://www.lillo.org.ar/phylogeny/tnt/ZIPCHTNT.ZIP', "$($TNT_PATH)\ZIPCHTNT.ZIP")

write-host "`r`nUnpacking..."
Expand-Archive -Path "$($TNT_PATH)\ZIPCHTNT.ZIP" -DestinationPath $TNT_PATH

write-host "`r`nAdding TNT to PATH"
$oldpath = (Get-ItemProperty -Path 'Registry::HKEY_LOCAL_MACHINE\System\CurrentControlSet\Control\Session Manager\Environment' -Name PATH).path
$newpath = "$oldpath;$TNT_PATH;"
Set-ItemProperty -Path 'Registry::HKEY_LOCAL_MACHINE\System\CurrentControlSet\Control\Session Manager\Environment' -Name PATH -Value $newpath

write-host "`r`nSystem old PATH:`r`n$oldpath"
write-host "`r`nSystem new PATH:`r`n$newpath"
write-host "In case of unexpected error with PATH. Use old PATH as your backup"

write-host "`r`nInstallation was completed. Write 'tnt' command in new terminal to start the program."
write-host "`r`nYou may need to restart your computer."
write-host "`r`If 'tnt' command is not working. Try to paste TNT path manually to PATH environment variable and check 'tnt' command in a new terminal."

