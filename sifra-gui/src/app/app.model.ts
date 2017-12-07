import { Component, OnInit, ViewChild } from '@angular/core';
import { ClassMetadataService } from './class-metadata.service';
import { ElementChooserComponent } from './element-chooser/element-chooser.component';

@Component({
    selector: 'app-root',
    template: `

    `
})
export class AppModel implements OnInit {
    // The existing components (at this stage, response models)
    public models: any = [];

    // The previously defined component to display.
    public currentComponent: any = null;

    // Is the save dialog visible?
    public showingSaveDialog: boolean = false;

    // Object containing the result to be saved.
    private resultObject: any = null;

    // Strings that represent the category. These are kept in lists due to the
    // interface of ng_select... ugly. There has to be a better way, but I did
    // not spend much time looking for it.
    private hazard: string[] = [];
    private sector: string[] = [];
    private facility: string[] = [];
    private component: string[] = [];

    // The name to save the component as.
    private name: string = null;

    // The description of the component.
    private description: string = null;

    // Has the current object been modified but not saved?
    private dirty: boolean = false;

    // 'Handle' to the element chooser.
    @ViewChild('elementchooser') elementChooser: ElementChooserComponent;

    constructor(private classMetadataService: ClassMetadataService) {}

    ngOnInit() {
        // this.getModels();
        // get the existing components (which are used to populate the list),
        this.classMetadataService.getCurrentModels()
            .subscribe(
                //components => this.models = models,
                error => alert(error)
            );
    }


}
