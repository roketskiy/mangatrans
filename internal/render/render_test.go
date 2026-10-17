package render

import (
 "image"
 "image/color"
 "image/png"
 "os"
 "path/filepath"
 "testing"
)

func TestRender(t *testing.T) {
 const font=`C:\Windows\Fonts\msyh.ttc`
 if _,err:=os.Stat(font); err!=nil { t.Skip("Microsoft YaHei font is not installed") }
 bg:=color.RGBA{180,180,180,255}; page:=image.NewRGBA(image.Rect(0,0,400,600)); for y:=0;y<600;y++ { for x:=0;x<400;x++ { page.SetRGBA(x,y,bg) } }
 boxes:=[]Block{{Box:image.Rect(270,30,390,300),Text:"你好世界测试文字",Vertical:true},{Box:image.Rect(10,350,190,570),Text:"横排测试"}}
 out,err:=Render(page,boxes,Options{FontPath:font,FontIndex:0,MinSize:2,MaxSize:14,Padding:8,TextColor:color.RGBA{0,0,0,255}}); if err!=nil { t.Fatal(err) }
 if out.Bounds().Size()!=page.Bounds().Size() { t.Fatalf("size = %v, want %v",out.Bounds().Size(),page.Bounds().Size()) }
 for i,b:=range boxes { found:=false; for y:=b.Box.Min.Y;y<b.Box.Max.Y;y++ { for x:=b.Box.Min.X;x<b.Box.Max.X;x++ { if out.At(x,y)!=bg { found=true; break } }; if found { break } }; if !found { t.Errorf("block %d has no rendered pixels",i) } }
 for _,p:=range []image.Point{{0,0},{399,0},{0,599},{399,599}} { if out.At(p.X,p.Y)!=bg { t.Errorf("corner %v changed",p) } }
 path:=filepath.Join("..","..",".cache","render-out","test.png"); if err:=os.MkdirAll(filepath.Dir(path),0755); err!=nil { t.Fatal(err) }; f,err:=os.Create(path); if err!=nil { t.Fatal(err) }; defer f.Close(); if err:=png.Encode(f,out); err!=nil { t.Fatal(err) }
}
