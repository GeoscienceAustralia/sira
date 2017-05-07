import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { HttpModule } from '@angular/http';
import { SelectModule } from 'ng2-select';

import { NDV_DIRECTIVES } from './click-to-edit/components';

import { AppComponent } from './app.component';
import { ElementEditorComponent } from './element-editor/element-editor.component';
import { ClassMetadataService } from './class-metadata.service';
import { StringDisplayComponent } from './element-editor/string-display/string-display.component';
import { PairListDisplayComponent } from './element-editor/pair-list-display/pair-list-display.component';
import { ElementChooserComponent } from './element-chooser/element-chooser.component';
import { DictDisplayComponent } from './element-editor/dict-display/dict-display.component';
import { SimpleDialogComponent } from './simple-dialog/simple-dialog.component';

@NgModule({
  declarations: [
    AppComponent,
    ElementEditorComponent,
    NDV_DIRECTIVES,
    StringDisplayComponent,
    PairListDisplayComponent,
    ElementChooserComponent,
    DictDisplayComponent,
    SimpleDialogComponent
  ],
  imports: [
    BrowserModule,
    FormsModule,
    HttpModule,
    SelectModule],
  providers: [ClassMetadataService],
  bootstrap: [AppComponent]
})
export class AppModule { }
