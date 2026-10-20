param(
    [string]$DocumentPath = (Join-Path $PSScriptRoot 'Supplementary_IL10_updated_20261020.docx'),
    [string]$OutputDirectory = (Join-Path $PSScriptRoot '../tmp/supplementary_render'),
    [Parameter(Mandatory=$true)][string]$PdfToPpm
)
$ErrorActionPreference = 'Stop'
$source = (Resolve-Path -LiteralPath $DocumentPath).Path
$destination = [System.IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Path $destination -Force | Out-Null
$pdf = Join-Path $destination 'supplementary.pdf'
$word = $null
$document = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $word.AutomationSecurity = 3
    $document = $word.Documents.Open($source, $false, $true, $false)
    $document.ExportAsFixedFormat($pdf, 17)
} finally {
    if ($document) { $document.Close(0); [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($document) }
    if ($word) { $word.Quit(); [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($word) }
}
& $PdfToPpm -r 120 -png $pdf (Join-Path $destination 'page')
if ($LASTEXITCODE -ne 0) { throw 'PDF page rendering failed' }
Write-Output "Rendered $source to $destination"
