import { Component, OnInit, ViewChild } from '@angular/core';
import { ClassMetadataService } from './class-metadata.service';
import { ElementChooserComponent } from './element-chooser/element-chooser.component';

enum HazardLevel {
    HAZARD,
    SECTOR,
    FACILITY,
    COMPONENT
}

@Component({
    selector: 'app-root',
    template: `
    <element-chooser
        #elementchooser
        [checkCanChange]="makeDirtyChecker()"
        (change)="chosenChanged($event)"
        (publish)="doPublish($event)">
    </element-chooser>
    <element-editor *ngIf="currentComponent"
        [value]="currentComponent"
        (publish)="doPublish($event)">
    </element-editor>

    <simple-dialog [(visible)]="showingSaveDialog" [closable]="false">
        <div *ngIf="levels">
            <table>
            <tr>
            <td><label>Hazard:</label></td>
            <td><ng-select
                [items]="levels.hazards"
                [active]="hazard"
                (selected)="selected($event, hazardLevel.HAZARD)"
                placeholder="please choose a hazard">
            </ng-select></td>
            </tr><tr>
            <td><label>Sector:</label></td>
            <td><ng-select
                [items]="levels.sectors"
                [active]="sector"
                (selected)="selected($event, hazardLevel.SECTOR)"
                placeholder="please choose a sector">
            </ng-select></td>
            </tr><tr>
            <td><label>Facility:</label></td>
            <td><ng-select
                [items]="levels.facilities"
                [active]="facility"
                (selected)="selected($event, hazardLevel.FACILITY)"
                placeholder="please choose a facility type">
            </ng-select></td>
            </tr><tr>
            <td><label>Component:</label></td>
            <td><ng-select
                [items]="levels.components"
                [active]="component"
                (selected)="selected($event, hazardLevel.COMPONENT)"
                placeholder="please choose a component type">
            </ng-select></td>
            </tr><tr>
            <td><label>Name:</label></td>
            <td><input type="text" [value]="name" (change)="setName($event)" style="width: 100%"><br></td>
            </tr><tr>
            <td><label>Description:</label><br></td>
            <td><textarea
                rows="4" cols="50"
                [value]="description"
                (change)="setDescription($event)">
            </textarea></td>
            </tr>
            </table>

            <button (click)="save()">Save</button>
            <button (click)="resetSave()">Reset</button>
            <button (click)="showingSaveDialog=false">Cancel</button>
        </div>
    </simple-dialog>

    <br/>
    <button (click)="showingSaveDialog=anythingToSave()">Save</button>
    <button (click)="reset()">Reset</button>
    <br/>

    <div *ngIf="components.length">
        <ul>
            <li *ngFor="let c of components" (click)="showComponent(c.id)">{{c.name}}</li>
        </ul>
    </div>
    `
})
export class AppComponent implements OnInit {
    // unfortunately globals are not visible within templates...
    public hazardLevel = HazardLevel;

    // A dictionary of lists describing the available categories for each level.
    public levels: any = null;

    // The existing components (at this stage, response models)
    public components: any = [];

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

    // The id of the previously defined component to display.
    private currentComponentId: string = null;

    // Has the current object been modified but not saved?
    private dirty: boolean = false;

    // 'Handle' to the element chooser.
    @ViewChild('elementchooser') elementChooser: ElementChooserComponent;

    constructor(private classMetadataService: ClassMetadataService) {}

    ngOnInit() {
        this.getLevels();
        // get the existing components (which are used to populate the list),
        this.classMetadataService.getInstancesOf('sifra.modelling.elements.ResponseModel')
            .subscribe(
                components => this.components = components,
                error => alert(error)
            );
    }

    makeDirtyChecker() {
        // Returns a callable that can be used to check if 'it is OK to reset'.
        // We need to do this because the scope of the call to the returned
        // callable is not the current scope.
        let _this = this;
        return () => !_this.dirty;
    }

    chosenChanged($event) {
        // Here we call with false as we do not want to reset the child element
        // (where the call originated from).
        this.reset(false);
    }

    doPublish($event) {
        this.resultObject = $event;
        this.dirty = true;
    }

    selected($event, level) {
        // One of the selects which define the category have been changed.
        switch(level) {
            case HazardLevel.HAZARD:
                this.hazard = [$event.id];
                break;
            case HazardLevel.SECTOR:
                this.sector = [$event.id];
                break;
            case HazardLevel.FACILITY:
                this.facility = [$event.id];
                break;
            case HazardLevel.COMPONENT:
                this.component = [$event.id];
                break;
        }
        // Update the lists describing the levels available.
        this.getLevels();
    }

    setName($event) {
        this.name = $event.target.value;
    }

    setDescription($event) {
        this.description = $event.target.value;
    }

    getLevels() {
        // Get the category lists specific to what has already been set.
        this.classMetadataService.getTypeLists(
            this.hazard[0], this.sector[0], this.facility[0], this.component[0]
        ).subscribe(
            levels => this.levels = levels,
            error => alert(error)
        );
    }

    // Check if there is anything to save.
    anythingToSave() {
        if(!this.resultObject) {
            alert('Nothing to save!');
            return false;
        }
        return true;
    }


    // Save a new instance.
    save() {
        if(!this.resultObject) {
            alert('No result to save!');
            return;
        }

        // The following checks should put notices on the inputs rather than
        // fire alerts.
        if(
            !this.hazard.length ||
            !this.sector.length ||
            !this.facility.length ||
            !this.component.length
        ) {
            alert('Please select a hazard, sector, facility and component type');
            return;
        }

        if(!this.name || !this.description) {
            alert('Please provide a name and description.');
            return;
        }

        // Add the sector to the result.
        this.resultObject['component_sector'] = {
            hazard: this.hazard[0],
            sector: this.sector[0],
            facility_type: this.facility[0],
            component: this.component[0]};

        // Add the name and description to the result.
        this.resultObject['attributes'] = {
            name: this.name,
            description: this.description};

        // Add the predecessor to the result.
        if(this.currentComponentId) {
            this.resultObject['predecessor'] = this.currentComponentId;
        }

        // Function for cleaning up after the save. Called whether or not the
        // save was successful.
        let cleanup = function(inst) {
            delete inst.resultObject['component_sector'];
            delete inst.resultObject['attributes'];
            delete inst.resultObject['predecessor'];
            inst.dirty = false;
            inst.showingSaveDialog=false;
        }

        // And finally (attempt to) save the result.
        this.classMetadataService.save(this.resultObject).subscribe(
            newComponent => {
                this.components.push(newComponent);
                this.currentComponentId = newComponent.id;
                cleanup(this);
            },
            error => {
                alert(error);
                cleanup(this);
            }
        );
    }

    // Reset the content of this element. Note that this does not reset the
    // metadata, which is done in `resetSave`.
    reset(resetChild = true) {
        this.currentComponent = null;
        this.currentComponentId = null;
        this.resultObject = null;
        this.dirty = false;
        if(resetChild) {
            this.elementChooser.reset();
        }
    }

    // Reset the metadata of the component.
    resetSave() {
        this.component = [];
        this.facility = [];
        this.hazard = [];
        this.sector = [];
        this.name = null;
        this.description = null;
        this.getLevels();
    }

    // When a component in the list of components is clicked, show that
    // component.
    showComponent(componentId: string) {
        if(this.dirty) {
            alert('Please save or discard current changes.');
            return;
        }

        this.reset();

        this.classMetadataService.getInstance(componentId)
            .subscribe(
                component => {
                    this.currentComponent = component;
                    this.currentComponentId = componentId;
                    this.resultObject = component._value;
                },
                error => alert(error)
            );
    }
}

