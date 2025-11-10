#!/bin/bash

rm -rf ./build
set -e

VERSION="0.1.3"
SOURCE="api.go"
LDFLAGS="-s -w -X main.buildTimestamp=$(date +%s)"
COMMON_TAGS="netgo"

OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

OUTDIR="build/$OS"
OUTPUT="$OUTDIR/antarys"

DEVELOPER_ID_CERT="Developer ID Application: Adib Mohsin (PLKDA3WBJJ)"
NOTARIZATION_PROFILE="notarytool-profile"
PKG_OUTPUT=""

mkdir -p "$OUTDIR"
export CGO_ENABLED=1

detect_cpu_features() {
    case "$ARCH" in
        x86_64)
            if [ -f /proc/cpuinfo ]; then
                cpu_features=$(grep "^flags" /proc/cpuinfo | head -1 | cut -d: -f2)
                echo "CPU Features: $(echo $cpu_features | tr ' ' '\n' | grep -E '(avx|fma|sse|aes)' | tr '\n' ' ')"
            fi
            ;;
        arm64 | aarch64)
            if [[ "$OS" == "darwin" ]]; then
                echo "Apple Silicon Features: Native M-series optimization"
            elif [ -f /proc/cpuinfo ]; then
                cpu_features=$(grep "^Features" /proc/cpuinfo | head -1 | cut -d: -f2)
                echo "ARM Features: $(echo $cpu_features | tr ' ' '\n' | grep -E '(neon|asimd|crc32|aes)' | tr '\n' ' ')"
            fi
            ;;
    esac
}

determine_cpu_flags() {
    detect_cpu_features

    local base_flags="-O3 -ffast-math -funroll-loops -finline-functions -fomit-frame-pointer"
    local vector_flags=""
    local math_flags=""
    local lto_flags="-flto=auto"

    case "$ARCH" in
        x86_64)
            math_flags="-mfpmath=sse -msse4.2 -fno-math-errno -ffinite-math-only -freciprocal-math"
            if grep -q "AuthenticAMD" /proc/cpuinfo 2>/dev/null; then
                if grep -q "avx512" /proc/cpuinfo 2>/dev/null; then
                    vector_flags="-march=znver4 -mtune=znver4 -mavx512f -mavx512bw -mavx512vl -mavx512dq"
                    base_flags="$base_flags -mprefer-vector-width=512"
                elif grep -q "avx2" /proc/cpuinfo 2>/dev/null; then
                    vector_flags="-march=znver3 -mtune=znver3 -mavx2 -mfma -msha -maes"
                    base_flags="$base_flags -mprefer-vector-width=256"
                else
                    vector_flags="-march=znver2 -mtune=znver2 -mavx2 -mfma"
                    base_flags="$base_flags -mprefer-vector-width=256"
                fi
            elif grep -q "GenuineIntel" /proc/cpuinfo 2>/dev/null; then
                if grep -q "avx512" /proc/cpuinfo 2>/dev/null; then
                    vector_flags="-march=skylake-avx512 -mtune=skylake-avx512 -mavx512f -mavx512bw -mavx512vl"
                    base_flags="$base_flags -mprefer-vector-width=512"
                elif grep -q "avx2" /proc/cpuinfo 2>/dev/null; then
                    vector_flags="-march=haswell -mtune=haswell -mavx2 -mfma -maes"
                    base_flags="$base_flags -mprefer-vector-width=256"
                else
                    vector_flags="-march=core2 -mtune=core2 -msse4.2"
                    base_flags="$base_flags -mprefer-vector-width=128"
                fi
            else
                vector_flags="-march=x86-64-v3 -mtune=generic -mavx2 -mfma"
                base_flags="$base_flags -mprefer-vector-width=256"
            fi
            vector_flags="$vector_flags -ftree-vectorize -fvect-cost-model=dynamic"
            ;;
        arm64 | aarch64)
            math_flags="-fno-math-errno -ffinite-math-only -freciprocal-math"
            if [[ "$OS" == "darwin" ]]; then
                vector_flags="-mcpu=apple-m1 -mtune=apple-m1"
                base_flags="$base_flags -fvectorize -fslp-vectorize -ftree-vectorize"
            else
                if grep -q "neon\|asimd" /proc/cpuinfo 2>/dev/null; then
                    vector_flags="-mcpu=native -march=armv8.4-a+crypto+fp16+dotprod"
                    base_flags="$base_flags -fvectorize -fslp-vectorize -ftree-vectorize"
                else
                    vector_flags="-mcpu=cortex-a76 -march=armv8.2-a+crypto+fp16"
                    base_flags="$base_flags -fvectorize -ftree-vectorize"
                fi
            fi
            ;;
    esac

    export CGO_CFLAGS="$base_flags $vector_flags $math_flags $lto_flags"

    if [[ "$OS" == "darwin" ]]; then
        export CGO_LDFLAGS="-O3 $lto_flags -Wl,-dead_strip"
    else
        export CGO_LDFLAGS="-O3 $lto_flags -Wl,--gc-sections"
    fi
}

determine_go_flags() {
    local go_optimization="-gcflags=-B,-C,-d=checkptr=0,-d=nil.check=0,-l=4"

    case "$ARCH" in
        x86_64|arm64|aarch64)
            go_optimization="$go_optimization -gcflags=-d=ssa/check/off"
            ;;
    esac

    case "$OS" in
        darwin)
            BUILD_FLAGS="-trimpath -mod=readonly $go_optimization"
            ;;
        linux)
            BUILD_FLAGS="-trimpath -mod=readonly -buildmode=pie $go_optimization"
            export GOOS=linux
            ;;
        windows)
            BUILD_FLAGS="-trimpath -mod=readonly $go_optimization"
            OUTPUT="$OUTPUT.exe"
            ;;
        *)
            BUILD_FLAGS="-trimpath -mod=readonly $go_optimization"
            ;;
    esac

    COMMON_TAGS="$COMMON_TAGS,vector_optimized,simd_enabled"
}

optimize_runtime() {
    export GOGC=200
    export GOMAXPROCS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "4")
    export GOMEMLIMIT=0
}

sign_macos_binary() {
    if [[ "$OS" != "darwin" ]] || [ -z "$DEVELOPER_ID_CERT" ]; then
        return 0
    fi

    codesign --timestamp --options runtime --sign "$DEVELOPER_ID_CERT" "$OUTPUT"
    
    if [ $? -eq 0 ]; then
        echo "Code signing successful"
        codesign --verify --deep --strict --verbose=2 "$OUTPUT"
    else
        echo "Code signing failed"
        exit 1
    fi
}

create_macos_installer() {
    if [[ "$OS" != "darwin" ]] || [ -z "$DEVELOPER_ID_CERT" ]; then
        return 0
    fi

    echo "📦 Creating macOS Installer Package"

    local PKG_IDENTIFIER="com.antarys.database"
    local PKG_INSTALL_LOCATION="/usr/local/bin"
    local PKG_NAME="Antarys-VectorDB-$VERSION"
    
    local pkg_root="$OUTDIR/pkg_root"
    local pkg_scripts="$OUTDIR/pkg_scripts"
    local pkg_resources="$OUTDIR/pkg_resources"
    
    rm -rf "$pkg_root" "$pkg_scripts" "$pkg_resources"
    mkdir -p "$pkg_root$PKG_INSTALL_LOCATION"
    mkdir -p "$pkg_scripts"
    mkdir -p "$pkg_resources"
    
    cp "$OUTPUT" "$pkg_root$PKG_INSTALL_LOCATION/"
    chmod 755 "$pkg_root$PKG_INSTALL_LOCATION/$(basename "$OUTPUT")"
    
    cat > "$pkg_scripts/postinstall" << 'POSTINSTALL_EOF'
#!/bin/bash

if ! echo $PATH | grep -q "/usr/local/bin"; then
    if ! grep -q "^/usr/local/bin$" /etc/paths 2>/dev/null; then
        echo "/usr/local/bin" >> /etc/paths
    fi
fi

if [ ! -L "/usr/bin/antarys" ]; then
    ln -sf "/usr/local/bin/antarys" "/usr/bin/antarys" 2>/dev/null || true
fi

echo "Antarys Vector Database v$VERSION installed successfully!"
echo "Run 'antarys' from any terminal to get started."
echo "Documentation: https://docs.antarys.ai/"

exit 0
POSTINSTALL_EOF

    chmod 755 "$pkg_scripts/postinstall"
    
    cat > "$pkg_scripts/preinstall" << 'PREINSTALL_EOF'
#!/bin/bash

if [ -f "/usr/local/bin/antarys" ]; then
    rm -f "/usr/local/bin/antarys"
fi

if [ -L "/usr/bin/antarys" ]; then
    rm -f "/usr/bin/antarys"
fi

exit 0
PREINSTALL_EOF

    chmod 755 "$pkg_scripts/preinstall"
    
    cat > "$pkg_resources/README.txt" << README_EOF
Antarys Vector Database v$VERSION

High-performance vector database optimized for SIMD operations and modern CPU architectures.

Developer: Antarys AI
Website: https://antarys.ai/
Documentation: https://docs.antarys.ai/
Cookbook: https://docs.antarys.ai/docs/cookbook
Support: antarys.ai@gmail.com

This installer places the 'antarys' binary in /usr/local/bin and makes it available system-wide.

After installation, run 'antarys' from any terminal to get started.
README_EOF

    local component_pkg="$OUTDIR/$PKG_NAME-component.pkg"
    
    pkgbuild \
        --root "$pkg_root" \
        --scripts "$pkg_scripts" \
        --identifier "$PKG_IDENTIFIER" \
        --version "$VERSION" \
        --install-location "/" \
        "$component_pkg"
    
    if [ $? -ne 0 ]; then
        echo "Failed to build component package"
        return 1
    fi
    
    local distribution_xml="$OUTDIR/distribution.xml"
    cat > "$distribution_xml" << DIST_EOF
<?xml version="1.0" encoding="utf-8"?>
<installer-gui-script minSpecVersion="1">
    <title>Antarys Vector Database v$VERSION</title>
    <organization>com.antarys</organization>
    <domains enable_localSystem="true"/>
    <options customize="never" require-scripts="true" rootVolumeOnly="true" />
    
    <readme file="README.txt"/>
    
    <pkg-ref id="$PKG_IDENTIFIER"/>
    
    <choices-outline>
        <line choice="default">
            <line choice="$PKG_IDENTIFIER"/>
        </line>
    </choices-outline>
    
    <choice id="default"/>
    <choice id="$PKG_IDENTIFIER" visible="false">
        <pkg-ref id="$PKG_IDENTIFIER"/>
    </choice>
    
    <pkg-ref id="$PKG_IDENTIFIER" 
             version="$VERSION" 
             onConclusion="none">$(basename "$component_pkg")</pkg-ref>

</installer-gui-script>
DIST_EOF

    local final_pkg="$OUTDIR/$PKG_NAME.pkg"
    
    productbuild \
        --distribution "$distribution_xml" \
        --resources "$pkg_resources" \
        --package-path "$OUTDIR" \
        "$final_pkg"
    
    if [ $? -ne 0 ]; then
        echo "Failed to build distribution package"
        return 1
    fi
    
    local signed_pkg="$OUTDIR/$PKG_NAME-signed.pkg"
    local installer_cert=$(security find-identity -v -p basic | grep "Developer ID Installer" | head -1 | sed -n 's/.*"\(.*\)".*/\1/p')
    
    if [ -z "$installer_cert" ]; then
        echo "No 'Developer ID Installer' certificate found"
        cp "$final_pkg" "$signed_pkg"
    else
        productsign \
            --sign "$installer_cert" \
            "$final_pkg" \
            "$signed_pkg"
        
        if [ $? -eq 0 ]; then
            echo "Package signed successfully"
            pkgutil --check-signature "$signed_pkg"
        else
            echo "Package signing failed"
            cp "$final_pkg" "$signed_pkg"
        fi
    fi
    
    rm -rf "$pkg_root" "$pkg_scripts" "$pkg_resources" "$component_pkg" "$final_pkg" "$distribution_xml"
    
    PKG_OUTPUT="$signed_pkg"
    
    echo "macOS Installer Package ready: $PKG_OUTPUT"
    ls -lh "$PKG_OUTPUT"
    
    return 0
}

notarize_macos_package() {
    if [[ "$OS" != "darwin" ]] || [ -z "$PKG_OUTPUT" ] || [ -z "$NOTARIZATION_PROFILE" ]; then
        return 0
    fi

    echo "Package Notarization"

    local submission_result=$(xcrun notarytool submit --wait --no-progress \
        --keychain-profile "$NOTARIZATION_PROFILE" \
        "$PKG_OUTPUT" 2>&1)
    
    if echo "$submission_result" | grep -q "status: Accepted"; then
        echo "Package notarization successful"
        
        xcrun stapler staple "$PKG_OUTPUT"
        
        if [ $? -eq 0 ]; then
            echo "Notarization ticket stapled"
        fi
        
        if spctl -a -t install -v "$PKG_OUTPUT" 2>/dev/null; then
            echo "Package passes GateKeeper verification"
        fi
        
        echo "Notarized installer ready for distribution"
        echo "$PKG_OUTPUT"
        
    else
        echo "Package notarization failed"
        local submission_id=$(echo "$submission_result" | grep -o "id: [a-f0-9-]*" | head -1 | cut -d' ' -f2)
        
        if [ ! -z "$submission_id" ]; then
            xcrun notarytool log "$submission_id" --keychain-profile "$NOTARIZATION_PROFILE"
        fi
        
        return 1
    fi
}

main() {
    echo "=== Antarys Vector Database v$VERSION - Optimized Build ==="
    echo "OS: $OS | Architecture: $ARCH"

    determine_cpu_flags
    determine_go_flags
    optimize_runtime

    echo ""
    echo "Building..."
    
    time go build \
        -ldflags="$LDFLAGS" \
        $BUILD_FLAGS \
        -tags "$COMMON_TAGS" \
        -o "$OUTPUT" \
        "$SOURCE"

    if [ $? -eq 0 ]; then
        echo "Build successful"
        
        sign_macos_binary
        create_macos_installer
        notarize_macos_package
        
        if [[ "$OS" != "darwin" ]] && command -v upx >/dev/null 2>&1; then
            read -p "🗜️ Compress with UPX? (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                upx --best "$OUTPUT" 2>/dev/null || upx -9 "$OUTPUT"
            fi
        fi
        
        echo ""
        echo "Antarys compiled with v$VERSION"
        echo "🌐 https://antarys.ai/"

    else
        echo "Build failed"
        exit 1
    fi
}

main "$@"